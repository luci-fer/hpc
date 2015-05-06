#include <cv.h>
#include <highgui.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <fstream>

#define RED 2
#define GREEN 1
#define BLUE 0

#define Mask_Width 3
#define Tile_Size 32

__constant__ char MC[Mask_Width*Mask_Width];

using namespace cv;

__device__ unsigned char Clamp(int dato){
    if(dato < 0)
        dato = 0;
    else
        if(dato > 255)
            dato = 255;
    return (unsigned char)dato;
}

__global__ void img2gray(unsigned char *imageIn, int width, int height, unsigned char *imageOut){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOut[row*width+col] = imageIn[(row*width+col)*3+RED]*0.30 + imageIn[(row*width+col)*3+GREEN]*0.59 + imageIn[(row*width+col)*3+BLUE]*0.11;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void CacheSobel(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * MC[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = Clamp(Pvalue);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void GlobalSobel(unsigned char *imageIn, int width, int height, unsigned int maskWidth, char *M , unsigned char *imageOut){
    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Valor = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Valor += imageIn[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOut[row*width+col] = Clamp(Valor);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void SharedSobel(unsigned char *imageInput, int width, int height, \
    unsigned int maskWidth,unsigned char *imageOutput){
    __shared__ float N_ds[Tile_Size + Mask_Width - 1][Tile_Size+ Mask_Width - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*Tile_Size+threadIdx.x, destY = dest / (Tile_Size+Mask_Width-1), destX = dest % (Tile_Size+Mask_Width-1),
        srcY = blockIdx.y * Tile_Size + destY - n, srcX = blockIdx.x * Tile_Size + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * Tile_Size + threadIdx.x + Tile_Size * Tile_Size;
    destY = dest /(Tile_Size + Mask_Width - 1), destX = dest % (Tile_Size + Mask_Width - 1);
    srcY = blockIdx.y * Tile_Size + destY - n;
    srcX = blockIdx.x * Tile_Size + destX - n;
    src = (srcY * width + srcX);
    if (destY < Tile_Size + Mask_Width - 1) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int accum = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * MC[y * maskWidth + x];
    y = blockIdx.y * Tile_Size + threadIdx.y;
    x = blockIdx.x * Tile_Size + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = Clamp(accum);
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){
    cudaSetDevice(0);
    const char imageIn[20] = "./inputs/img2.jpg";
    char h_M[] = {-1,0,1,-2,0,2,-1,0,1}, *d_M;
    unsigned char *dataImage, *d_dataImage, *d_imageOut;
    unsigned char *h_OutC, *h_OutG, *h_OutS;
    unsigned char *d_SobelG, *d_SobelC, *d_SobelS;
    clock_t startCPU, endCPU, startGlobal, endGlobal, startCache, endCache, startShared, endShared, startCopy, endCopy, endCacheCopy;
    double Time_Global, Time_Cache, Time_Shared, Time_CPU;
    cudaError_t error = cudaSuccess;
    
    Mat image;
    image = imread(&imageIn[0], 1);

    if(!image.data){
        printf("No image Data \n");
        return -1;
    }

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;
   
    dataImage = (unsigned char*)malloc(size);
    h_OutG = (unsigned char *)malloc(sizeGray);
    h_OutC = (unsigned char *)malloc(sizeGray);
    h_OutS = (unsigned char *)malloc(sizeGray);
  
    error = cudaMalloc((void**)&d_dataImage, size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataImage\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_imageOut, sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOut\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_M,sizeof(char)*9);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_SobelG, sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelG\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_SobelC, sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelC\n");
        exit(-1);
    }

    error = cudaMalloc((void**)&d_SobelS, sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelS\n");
        exit(-1);
    }
  
    dataImage = image.data;  //------------------------> POTENCIAL ERROR!
      
//////////////////////////////////////////////////////OPENCV////////////////////////////////////////////

    startCPU = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    endCPU = clock();
    Time_CPU = ((double) (endCPU - startCPU)) /CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo con OPENCV: %.10f\n", Time_CPU);
/////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////COPY TIME////////////////////////////////////////////////
    startCopy = clock();

    int blockSize = Tile_Size;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);

    error = cudaMemcpy(d_dataImage, dataImage, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }

    endCacheCopy = clock();

    error = cudaMemcpy(d_M, h_M, sizeof(char)*9, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_M a d_M \n");
        exit(-1);
    }

    img2gray <<<dimGrid,dimBlock>>> (d_dataImage, width, height, d_imageOut);

    endCopy = clock();
//////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////GLOBAL MEMORY//////////////////////////////////////////////
    startGlobal = clock();

    cudaDeviceSynchronize();
    GlobalSobel <<<dimGrid,dimBlock>>> (d_imageOut, width, height, 3, d_M, d_SobelG);
    cudaMemcpy(h_OutG, d_SobelG, sizeGray, cudaMemcpyDeviceToHost);

    endGlobal = clock();

    Time_Global = ((double) (endGlobal - startGlobal) + (endCopy - startCopy))  / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo con Global Memory: %.10f\n", Time_Global);

/////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////CACHE MEMORY//////////////////////////////////////////////
    startCache = clock();

    cudaMemcpyToSymbol (MC, h_M, Mask_Width*sizeof(char));

    cudaDeviceSynchronize();
    CacheSobel <<<dimGrid,dimBlock>>> (d_imageOut, width, height, 3, d_SobelC);
    cudaMemcpy(h_OutC, d_SobelC, sizeGray, cudaMemcpyDeviceToHost);

    endCache = clock();

    Time_Cache = ((double) (endCache - startCache) + (endCacheCopy - startCopy)) / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo con Cache Memory: %.10f\n", Time_Cache);

/////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////SHARED MEMORY//////////////////////////////////////////////
    startShared = clock();

    cudaDeviceSynchronize();
    SharedSobel <<<dimGrid,dimBlock>>> (d_imageOut, width, height, 3, d_SobelS);
    cudaMemcpy(h_OutS, d_SobelS, sizeGray, cudaMemcpyDeviceToHost);

    endShared = clock();

    Time_Shared = ((double) (endShared - startShared) + (endCopy - startCopy))  / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo con Shared Memory: %.10f\n", Time_Shared);

/////////////////////////////////////////////////////////////////////////////////////////////////////

    Mat gray_image_global;
    gray_image_global.create(height, width, CV_8UC1);
    gray_image_global.data = h_OutG;

    Mat gray_image_cache;
    gray_image_cache.create(height, width, CV_8UC1);
    gray_image_cache.data = h_OutC;

    Mat gray_image_shared;
    gray_image_shared.create(height, width, CV_8UC1);
    gray_image_shared.data = h_OutS;

    //imwrite("./Sobel_Image_Global.jpg", gray_image_global);
    //imwrite("./Sobel_Image_Cache.jpg", gray_image_cache);
    //imwrite("./Sobel_Image_Shared.jpg", gray_image_shared);

    //namedWindow(imageName, WINDOW_NORMAL);
    //namedWindow("Gray Image CUDA GLOBAL", WINDOW_NORMAL);
    //namedWindow("Gray Image CUDA CACHE", WINDOW_NORMAL);
    //namedWindow("Gray Image CUDA SHARED", WINDOW_NORMAL);
    //namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);

    //imshow(imageName, image);
    //imshow("Gray Image CUDA GLOBAL", gray_image_global);
    //imshow("Gray Image CUDA CACHE", gray_image_cache);
    //imshow("Gray Image CUDA SHARED", gray_image_shared);
    //imshow("Sobel Image OpenCV", abs_grad_x);

    //waitKey(0);

    //free(dataImage);
    free(h_OutG);
    free(h_OutC);
    free(h_OutS);
    cudaFree(d_dataImage);
    cudaFree(d_imageOut);
    cudaFree(d_M);
    cudaFree(d_SobelG);
    cudaFree(d_SobelC);
    cudaFree(d_SobelS);
    return 0;
}