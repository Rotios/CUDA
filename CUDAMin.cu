/*
  @ 2017 Jose Rivas-Garcia and John Freeman
  The code creates n*n blocks, which perform the n*n comparisons.
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define ARRAYSIZE 1024

__global__ void minCompare(int *a, bool *check) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx == idy) { return; }

    int xval = a[idx];
    int yval = a[idy];

    if (xval > yval) {
        check[idx] = false;
    }
}

__global__ void cudaMin(int *a, bool *check, int* min) {
    int idx = blockIdx.x;

    if (check[idx]) {
        min[0] = a[idx];
    }
}

__global__ void cudaBoolFill(bool *arr, int length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        arr[i] = true;
    }
}

void array_fill(int *arr, int length)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = (int)rand();
    }
}

int findMin(int *arr, const int length){
    bool *check;
    int *ad, *min;
    
    const int intSize = sizeof(int);
    const int asize = length * intSize;
    const int bsize = length * sizeof(bool);
    
    cudaMalloc((void**)&ad, asize);
    cudaMemcpy(ad, arr, asize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&check, bsize);
    cudaBoolFill <<< dim3(length, 1), 1 >>> (check, length);

    cudaMalloc((void**)&min, intSize);

    minCompare <<< dim3(length, length), 1 >>> (ad, check);
    cudaMin <<< dim3(length, 1), 1 >>> (ad, check, min);

    int minhost[1];
    cudaMemcpy(minhost, min, intSize, cudaMemcpyDeviceToHost);

    cudaFree(ad);
    cudaFree(min);
    cudaFree(check);

    return minhost[0];
}

int main()
{
    int *a = (int*)malloc(ARRAYSIZE * sizeof(int));
    array_fill(a, ARRAYSIZE);
    
    clock_t start = clock();

    int min = findMin(a, ARRAYSIZE);

    clock_t stop = clock();
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);

    printf("min is %d\n", min);

    free(a);
    while (1) {}
    return 0;
}
