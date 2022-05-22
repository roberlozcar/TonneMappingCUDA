#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}


// Unroll warp min
__device__ void warpreducemin(volatile float* mindata, unsigned int tid)
{
    mindata[tid] = min(mindata[tid], mindata[tid + 32]);
    mindata[tid] = min(mindata[tid], mindata[tid + 16]);
    mindata[tid] = min(mindata[tid], mindata[tid + 8]);
    mindata[tid] = min(mindata[tid], mindata[tid + 4]);
    mindata[tid] = min(mindata[tid], mindata[tid + 2]);
    mindata[tid] = min(mindata[tid], mindata[tid + 1]);
}

// Unroll warp max
__device__ void warpreducemax( volatile float* maxdata, unsigned int tid){
    maxdata[tid] = max(maxdata[tid], maxdata[tid + 32]);
    maxdata[tid] = max(maxdata[tid], maxdata[tid + 16]);
    maxdata[tid] = max(maxdata[tid], maxdata[tid + 8]);
    maxdata[tid] = max(maxdata[tid], maxdata[tid + 4]);
    maxdata[tid] = max(maxdata[tid], maxdata[tid + 2]);
    maxdata[tid] = max(maxdata[tid], maxdata[tid + 1]);
}


// Reducción al minimo
__global__ void reducemin(const float* const d_logLuminance, const int l, float* outmin) {
    int i = threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float minvec[];

    // Copiamos datos a shared mem
    if (j > l) {
        minvec[i] = d_logLuminance[i]; // Se le asocia un valor de cualquiera del array, no afecta a la operacion min o max
    }
    else {
        minvec[i] = d_logLuminance[j];
    }
    __syncthreads();
    
    // Se va reduciendo el array por la mitad en cada paso
    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (i < s) {
            minvec[i] = min(minvec[i], minvec[i + s]);
        }
        __syncthreads();
    }

    // Se desenrrolla el ultimo warp
    if (i < 32) warpreducemin(minvec, i);

    //Se copia el valor a la salida
    if (i == 0) {
        outmin[blockIdx.x] = minvec[0];
    }
}


// Reducción al maximo
__global__ void reducemax(const float* const d_logLuminance, const int l, float* outmax) {
    int i = threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float maxvec[];
    if (j > l) {
        maxvec[i] = d_logLuminance[i];
    }
    else {
        maxvec[i] = d_logLuminance[j];
    }
    __syncthreads();


    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if (i < s) {
            maxvec[i] = max(maxvec[i], maxvec[i + s]);
        }
        __syncthreads();
    }

    if (i < 32) warpreducemax(maxvec, i);

    if (i == 0) {
        outmax[blockIdx.x] = maxvec[0];
    }
}

__global__ void histogram(const float* const d_logLuminance, const float minimum, 
    const float range, const int nbins, const int l,unsigned int * hist) {
    
    int i= blockIdx.x * blockDim.x + threadIdx.x;
    int bin;
    if (i < l) {
        // Calculamos el bin y lo sumamos a hist de forma atomica
        bin = (d_logLuminance[i] - minimum) / range * nbins;
        atomicAdd(&(hist[bin]), 1);
    }
}

__global__ void exclusivescan(unsigned int* hist,unsigned int* d_cdf,const int numbins) {
    extern __shared__ int temp[];
    int i= threadIdx.x;
    int j= blockIdx.x * blockDim.x + threadIdx.x;
    int ai = i;
    int bi = i + numbins / 2;
    int offset = 1;
    int itemp;

    // Se inicializa la memoria compartida
    temp[ai] = hist[i];
    temp[bi] = hist[i + numbins / 2];

    for (int ii = numbins >> 1; ii > 0; ii >>= 1) {
        __syncthreads();
        if (i < ii) {
            // Se calculan los indices y se suman los elementos de dos en dos
            ai = offset * (2 * i + 1) - 1;
            bi = offset * (2 * i + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Se coloca el cero al final del array
    if (i == 0)temp[numbins - 1] = 0;

    for (int ii = 1; ii < numbins; ii <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (i < ii) {
            ai = offset * (2 * i + 1) - 1;
            bi = offset * (2 * i + 2) - 1;
            itemp = temp[ai];
            // Se va moviendo el cero y calculando las sumas que faltan
            temp[ai] = temp[bi];
            temp[bi] += itemp;
        }
    }

    __syncthreads();

    // Se copia a la salida los valores calculados
    d_cdf[j] = temp[i];
    d_cdf[j + numbins / 2] = temp[i + numbins / 2];
}


void calculate_cdf(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*
    1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 
	2) Obtener el rango a representar
	3) Generar un histograma de todos los valores del canal logLuminance usando la formula 
	bin = (Lum [i] - lumMin) / lumRange * numBins
	4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf) 
	de los valores de luminancia. Se almacenar en el puntero c_cdf
  */    

    // Minmax reduction
    int l = numCols * numRows;
    
    // Definimos los tamaños de bloque y grid para el primer algoritmo
    int blocknum = 1024;
    const dim3 blockSize(blocknum, 1, 1);
    dim3 gridSize((l - 1) / blocknum + 1, 1, 1);

    // Se definen e inicializan las variables que se van a usar en las reducciones
    float* ominvec, * omaxvec, * iminvec, * imaxvec;
    checkCudaErrors(cudaMalloc(&ominvec, sizeof(float) *gridSize.x));
    checkCudaErrors(cudaMalloc(&omaxvec, sizeof(float) *gridSize.x));
    checkCudaErrors(cudaMalloc(&iminvec, sizeof(float) * l));
    checkCudaErrors(cudaMalloc(&imaxvec, sizeof(float) * l));
    checkCudaErrors(cudaMemcpy(iminvec, d_logLuminance, sizeof(float) * l, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(imaxvec, d_logLuminance, sizeof(float) * l, cudaMemcpyDeviceToDevice));


    while (true) {
        reducemin << <gridSize, blockSize, blocknum*sizeof(float) >> > (iminvec, l, ominvec);
        cudaDeviceSynchronize();
        reducemax << <gridSize, blockSize, blocknum*sizeof(float) >> > (imaxvec, l, omaxvec);
       cudaDeviceSynchronize();
        if (gridSize.x == 1) break; //Obtenemos el resultado final
        //Copiamos la salida en la entrada para la siguiente aplicacion
        checkCudaErrors(cudaMemcpy(iminvec, ominvec, sizeof(float) * gridSize.x, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(imaxvec, omaxvec, sizeof(float) * gridSize.x, cudaMemcpyDeviceToDevice));
        // Reducimos el tamaño del grid y redondeamos hacia arriba si es menor que uno
        gridSize.x /= blocknum;
        if (gridSize.x == 0)gridSize.x++;
    }
    
    // Copiamos un float a las variables de min y max
    checkCudaErrors(cudaMemcpy(&min_logLum, ominvec, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max_logLum, omaxvec, sizeof(float), cudaMemcpyDeviceToHost));

    // Liberamos memoria
    checkCudaErrors(cudaFree(iminvec));
    checkCudaErrors(cudaFree(ominvec));
    checkCudaErrors(cudaFree(imaxvec));
    checkCudaErrors(cudaFree(omaxvec));


    // Histograma
    float lumrange = max_logLum - min_logLum;
    
    gridSize.x = (l - 1) / blocknum + 1;

    unsigned int* d_hist;
    checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMemset(d_hist, 0, numBins * sizeof(unsigned int)));
    
    histogram << <gridSize, blockSize >> > (d_logLuminance, min_logLum, lumrange, numBins,l,d_hist);

    // Distribucion acumulada
    gridSize.x = (numBins - 1) / blocknum + 1;
    exclusivescan << <gridSize, blockSize, numBins * sizeof(int) * 2 >> > (d_hist, d_cdf, numBins);
    checkCudaErrors(cudaGetLastError());

    // Liberamos recursos
    checkCudaErrors(cudaFree(d_hist));
}


