
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TILE 32



///////////////////////////////////////////FUNCION INICIALIZAR  LAS VECTORES//////////////////////////////////
//////////////////Todas la posiciones del vector son inicializadas con valores aleatorios/////////////////////
void inic_vector(int* vector, int tam){
  for(int i=0; i<tam; i++){
    vector[i] = rand()%101;
  }
}

/////////////////////////////////////SUMA DE ELEMENTOS DE UN VECTOR///////////////////////////////////
int sum_vectores(int* A, int tam){
  int suma=0;
  for (int i=0; i<tam; ++i){
    suma = suma + A[i];
    }
  return 0;
}

////////////////////////////KERNEL SUMATORIA DE VECTOR SIN TILING//////////////////////////
__global__ void vectorSumKernel(int *d_A, int tam){
    int vect = blockIdx.x*blockDim.x + threadIdx.x;

    if(vect < tam){
        for (int k = 0; vect < tam ; ++k){
            d_A[vect] = d_A[vect] + d_A[vect+(tam*k)];
        }
      }
}

////////////////////////////KERNEL SUMATORIA DE VECTOR CON TILING//////////////////////////
__global__ void vectorSumKernelTiled(int *d_A, int tam){

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int vect = bx * TILE + tx;

    int p = tam/TILE;
    
    if(vect < tam){
        for (int k = 0; vect < tam / TILE; ++k){
            d_A[vect] = d_A[vect] + d_A[vect + (p*k)];
            __syncthreads();
        }
    }
}



int main(){
  clock_t sec_ini, sec_fin, par_ini, par_fin, tile_ini, tile_fin;
  double tiempo_sec, tiempo_par, tiempo_tile;
   
  int tam= 1024;  
  
  
  //////////////////////////////VARIABLES EN HOST/////////////////////////////////
  int *h_A;
  ////////////////////////////////Reservar memoria///////////////////////////////
  h_A = (int * ) malloc (tam);
  /////////////////////////Inicializar variables en host/////////////////////////
  inic_vector(h_A, tam);
  ////////////////////////////VARIABLES EN HOST////////////////////////////////
   
  
  ///////////////////////////////REALIZAR MULTIPLICACION SECUENCIAL////////////////////////////
  
  sec_ini=clock();
  
  sum_vectores(h_A, tam);
  
  sec_fin= clock();
  tiempo_sec= ((double) (sec_fin - sec_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO SECUENCIAL TARDO: %.10f\n", tiempo_sec);
//////////////////////////////////////////////////////////////////////////////////////////// 
  

    
  
/////////////////////////////EJECUCUCION ALGORITMO PARALELO/////////////////////////////////////////
 /////////////////////////////Variables en device////////////////////////////
  int *d_A;
  cudaError_t error = cudaSuccess;
///////////////////////////////Reserva de memoria////////////////////////////////
  
  //cudaMalloc((void**)&d_A, tam);

  error = cudaMalloc((void**)&d_A,tam*2);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_A");
        exit(0);
    }
  
  //cudaMemcpy(d_A, h_A, tam,  cudaMemcpyHostToDevice);  

   error = cudaMemcpy(d_A, h_A, tam, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando datos a d_A");
        exit(0);
    }

  int blockSize = 32;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(tam/float(blockSize)),ceil(tam/float(blockSize)),1);
  
  
  par_ini=clock();
  vectorSumKernel <<<dimGrid,dimBlock>>> (d_A,tam);
  cudaDeviceSynchronize();
  cudaMemcpy(h_A,d_A,tam,cudaMemcpyDeviceToHost);
  
  
  par_fin=clock();
  tiempo_par= ((double) (par_fin - par_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO PARALELO TARDO: %.10f\n", tiempo_par);
    
  
  cudaFree(d_A);
  
  
  ///////////////////////////////////////////////////////////////////////////////
  
  
  tile_ini=clock();
  
      
  vectorSumKernelTiled<<<dimGrid,dimBlock>>>(d_A,tam);
  cudaDeviceSynchronize();
  cudaMemcpy(h_A,d_A,tam,cudaMemcpyDeviceToHost);
   
  
  tile_fin=clock();
  
  tiempo_tile= ((double) (tile_fin - tile_ini)) / CLOCKS_PER_SEC;
  printf("EL ALGORITMO PARALELO USANDO TILING TARDO: %.10f\n", tiempo_tile);
    
  
  cudaFree(d_A);
  
}