#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

__global__ void swap(int *a, const int arraySize, const int step, const int stage)
{
    int i = threadIdx.x;
    int listSize = 2 << step;
    int ij = i^stage;

    if (ij > i) {
        if ((i&listSize) == 0) {

            if (a[i] > a[ij]) {
                int temp = a[ij];
                a[ij] = a[i];
                a[i] = temp;
            }
        }
        else if ((i&listSize) != 0) {

            if (a[i] < a[ij]) {
                int temp = a[ij];
                a[ij] = a[i];
                a[i] = temp;
            }
        }
    }
}

__global__ void minCompare(int *a, bool *check){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if(idx == idy) { return; }
    
    int xval = a[idx];
    int yval = a[idy];

    if(xval > yval) {
        check[idx] = false;
    }
}

__global__ void cudaMin(int *a, bool *check, int* min) {
    //int idx = threadIdx.x;
    int idx = blockIdx.x;

    if(check[idx]) {
        min[0] = a[idx];
    }
}

void array_fill(int *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = (int) (float)rand();
  }
}

int main()
{
    const int arraySize = 1024;
    int *a = (int*) malloc( arraySize * sizeof(int));
    array_fill(a, arraySize);

    bool bools[arraySize];
    for (int k = 0; k < arraySize; ++k) {
        bools[k] = true;
    }

    bool *check;
    int *ad;

    const int asize = arraySize * sizeof(int);
    const int bsize = arraySize * sizeof(bool);

    cudaMalloc((void**) &check, bsize);
    cudaMemcpy(check, bools, bsize, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&ad, asize);
    cudaMemcpy(ad, a, asize, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(arraySize, arraySize);
    dim3 dimGrid(1, 1);
    dim3 minBlock(arraySize, 1);

    int *min;

    const int intSize = sizeof(int);

    cudaMalloc((void**) &min, intSize);
    
    clock_t start = clock();

    minCompare <<< dim3(arraySize, arraySize), 1 >>> (ad, check);
    cudaMin <<< dim3(arraySize, 1), 1 >>> (ad, check, min);
    
    clock_t stop = clock();
        double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);

    bool bools2[arraySize];
    int minhost[1];
    cudaMemcpy(minhost, min, intSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(bools2, check, bsize, cudaMemcpyDeviceToHost);
    printf("min is %d\n", minhost[0]);

      for (int k = 0; k < arraySize; ++k) {
        printf(bools2[k] ? "true " : "false ");
    }

     start = clock();
    //iterate through steps
    for (int i = 0; i < 10; ++i) {

        //iterate through stages
        for (int j = i; j >= 0; --j) {
            dim3 dimBlock2(arraySize, 1);
            int t = 1 << j;
            swap <<< dimGrid, dimBlock2 >>> (ad, arraySize, i, t);

        }
    }
     stop = clock();
    
     elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);


    int b[arraySize];
    cudaMemcpy(b, ad, asize, cudaMemcpyDeviceToHost);

    for (int k = 0; k < arraySize; ++k) {
        printf("%d ", b[k]);
    }
    
    printf("\n\n%d\n", a[0]);

    cudaFree(ad);
    while (1) {}
}
