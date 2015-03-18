#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32


using namespace std;

__global__ void multTiled(float *A, float *B, float *C,int rA,int cA,int rB,int cB){
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

     int Pvalue = 0;


  for(int k = 0; k < (cA+TILE_WIDTH-1)/(TILE_WIDTH); ++k){

    if(k*TILE_WIDTH + tx < cA && row < rA){
      Mds[ty][tx] = A[row*cA + k*TILE_WIDTH + tx];
    }else{
      Mds[ty][tx] = 0;
    }
    if(k*TILE_WIDTH + ty < cA && col < cB){
      Nds[ty][tx] = B[(k*TILE_WIDTH + ty) * cB + col];
    }else{
      Nds[ty][tx] =0;
    }

    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  if (row < rA && col < cB){
    C[row*cB+col] = Pvalue;
  }
}


__global__ void matrixMulKernel(float *d_A, float *d_B, float *d_C,int rA,int cA,int rB,int cB){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < rA)&&(col < cB)){
        Pvalue = 0;
        for (int k = 0; k < rB ; ++k){
            Pvalue += d_A[row*cA+k] * d_B[k*cB+col];
        }
        d_C[row*cB+col] = Pvalue;
    }
}



void multMxN(float *A,float *B, float *C, float rA,int cA,int rB,int cB){
	//El numero de col de A, debe ser igual al numero de row de B;
	//Matriz C= FilasAxColumnasB


	for(int i=0;i<rA;i++){
      	for(int j=0;j<cB;j++){
      		int value=0;
          	for(int k=0;k<rB;k++){
             	//h_c[n*i+j]+=h_a[n*i+k]*h_b[n*k+j];
             	value+=A[cA*i+k]*B[cB*k+j];
          	}
          	C[cB*i+j]=value;
      	}
    }
}

void llenar(float *MA,int n,int m){
	int tam=n*m;
	for(int  i = 0; i < tam; i++ )
        MA[i] = i;
}

void imprimir(float *MA,int n,int m){
	int tam=n*m,cont=0;
	for(int i = 0; i < tam; i++ ) {
        printf("%f ",MA[i]);
        if(cont==m-1){
            printf("\n");
            cont=-1;
        }
        cont++;  
    }
	printf("\n");
}

void comparar(float *C1,float *C2,int rC,int cC){

    int w=rC*cC;
    for(int i=0;i<w;i++){
      if(C1[i]!=C2[i]){
        cout<<"No son iguales"<<endl;
        break;
      }
    }
    cout<<"Son iguales"<<endl;
}

int main(){
	int rA=128;
	int cA=1024;
	int cB=512;
	float blockSize = 4;
  int rB=cA;

	size_t bytesA=(rA*cA)*sizeof(float);
	size_t bytesB=(rB*cB)*sizeof(float);
	size_t bytesC=(rA*cB)*sizeof(float);

	float *A=(float*)malloc(bytesA);
	float *B=(float*)malloc(bytesB);
	float *C1=(float*)malloc(bytesC);
	float *C2=(float*)malloc(bytesC);
  float *C3=(float*)malloc(bytesC);


	//1.Lleno las matrices
	llenar(A,rA,cA);
	llenar(B,rB,cB);

	//2.imprimo las matrices
	//cout<<"Matriz A:"<<endl;
	//imprimir(A,rA,cA);

	//cout<<"Matriz B:"<<endl;
	//imprimir(B,rB,cB);

	//3.Multiplico matrices NxM Secuencial
	clock_t start = clock();      
    multMxN(A,B,C1,rA,cA,rB,cB);
    clock_t end= clock(); 
  	double elapsed_seconds=end-start;
    printf("Tiempo transcurrido Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));
    //cout<<"Secuencial"<<endl;
	//imprimir(C1,rA,cB);
    
	//3.Multiplico matrices NxM Paralelo sin Tile
    float *d_A;
		float *d_B;
    float *d_C;
	// Allocate memory for each vector on GPU
    cudaMalloc(&d_A,bytesA);
    cudaMalloc(&d_B,bytesB);
    cudaMalloc(&d_C,bytesC);
	
    

	// Copy host vectors to device
    cudaMemcpy( d_A, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy( d_B, B, bytesB, cudaMemcpyHostToDevice);

  
	//bloques
	dim3 dimGrid(ceil(cB/blockSize),ceil(rA/blockSize),1);
  //hilos
  dim3 dimBlock(blockSize,blockSize,1);
  //Tiempo de ejecucion Paralelo
	clock_t start2 = clock();   
	 
    // Execute the kernel
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,rA,cA,rB,cB);
    //multTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,rA,cA,rB,cB);
    cudaDeviceSynchronize();
    // Copy array back to host
    cudaMemcpy( C2, d_C, bytesC, cudaMemcpyDeviceToHost );
    clock_t end2= clock(); 
  	double elapsed_seconds2=end2-start2;
    printf("Tiempo transcurrido Paralelo SinTile: %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));  
    //4.Imprimo matriz C
	//cout<<endl<<"Paralelo sin tiling"<<endl;
	//imprimir(C2,rA,cB);

  //cout<<"Matriz Secuencial-SinTiling: "<<endl;
  
 
	//cout<<endl;
  
  
   // Execute the kernel
     //Tiempo de ejecucion Paralelo
		clock_t start3 = clock();   
    multTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,rA,cA,rB,cB);
    cudaDeviceSynchronize();
    // Copy array back to host
    cudaMemcpy( C3, d_C, bytesC, cudaMemcpyDeviceToHost );
    clock_t end3= clock(); 
  	double elapsed_seconds3=end3-start3;
    printf("Tiempo transcurrido Paralelo ConTile: %lf\n", (elapsed_seconds3 / CLOCKS_PER_SEC));  
    //4.Imprimo matriz C
	//cout<<endl<<"Paralelo con tiling"<<endl;
	//imprimir(C3,rA,cB);
 // cout<<"Matriz Secuencial-Tiling: "<<endl;
  comparar(C1,C3,rA,cB);
	comparar(C1,C2,rA,cB);
	
	 // Release device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
 
    // Release host memory
    free(A);
    free(B);
    free(C1);
    free(C2);
	
	return 0;
}
