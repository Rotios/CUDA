#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define INITIAL_CAPACITY 1024

/******************** Find the min value **************************/
__global__ void minCompare(int *a, bool *check) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx == idy) { return; }

    int xval = a[idx];
    int yval = a[idy];
    
    if (xval == 0) {
        check[idx] = false;
    } else if (xval > yval) {
        check[idx] = false;
    }
}

__global__ void cudaMin(int *a, bool *check, int* min) {
    int idx = blockIdx.x;

    if (check[idx]) {
        min[0] = a[idx];
    }
}

/************************* Find the max value **********************/
__global__ void maxCompare(int *a, bool *check) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx == idy) { return; }

    int xval = a[idx];
    int yval = a[idy];

    if (xval < yval) {
        check[idx] = false;
    }
}

__global__ void cudaMax(int *a, bool *check, int* max) {
    int idx = blockIdx.x;

    if (check[idx]) {
        max[0] = a[idx];
    }
}

/*********************** Helper Methods ********************************************/
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

void findMin(int *arr, const int length, int& minimum) {
    bool *check;
    int *min;

    const int intSize = sizeof(int);
    const int asize = length * intSize;
    const int bsize = length * sizeof(bool);

    cudaMalloc((void**)&check, bsize);
    cudaBoolFill << < dim3(length, 1), 1 >> > (check, length);

    cudaMalloc((void**)&min, intSize);

    minCompare << < dim3(length, length), 1 >> > (arr, check);
    cudaMin << < dim3(length, 1), 1 >> > (arr, check, min);

    int minhost[1];
    cudaMemcpy(minhost, min, intSize, cudaMemcpyDeviceToHost);

    cudaFree(min);
    cudaFree(check);

    minimum = minhost[0];
}

int findMax(int *arr, const int length) {
    bool *check;
    int *max;

    const int intSize = sizeof(int);
    const int asize = length * intSize;
    const int bsize = length * sizeof(bool);

    cudaMalloc((void**)&check, bsize);
    cudaBoolFill << < dim3(length, 1), 1 >> > (check, length);

    cudaMalloc((void**)&max, intSize);

    minCompare << < dim3(length, length), 1 >> > (arr, check);
    cudaMin << < dim3(length, 1), 1 >> > (arr, check, min);

    int maxhost[1];
    cudaMemcpy(maxhost, max, intSize, cudaMemcpyDeviceToHost);

    cudaFree(max);
    cudaFree(check);

    return maxhost[0];
}

/********************* Find the Curl *****************************************/
__global__ void findCurl(int *sequence, int **table, int length, int *curl){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int finalIndex = length - 1;

    if(sequence[finalIndex] == sequence[finalIndex - (index + 1)]){
        table[finalIndex][index] = table[index][finalIndex - (index + 1)] + 1;
    } else {
        table[finalIndex][index] = 1;
    }
}

int findCurl2(int *sequence, int **table, int length){
    int *tempResults;
    cudaMalloc((void **) &tempResults, (length >> 1) * sizeof(int));

    for(int i(0); i < (length >> 1); ++i) {
        int *p = &(table[i][(length - 1) - i]);
        findMin(p, length, tempResults[i]);
    }
    int curl = findMax(tempResults, length);

    cudaFree(tempResults);

    return curl;
}

__global__ void fillInitialTable(int *sequence, int **table, int length, int seqPosition){
    int index = blockIdx.x;
    int finalIndex = length - 1;
    int checkIndex = seqPosition - (index + 1);

    if(index > seqPosition) {
        table[seqPosition][index] = 0;
    } else if(sequence[seqPosition] == sequence[seqPosition - (index )]){
        table[seqPosition][index] = table[index][seqPosition - (index )] + 1;
    } else {
        table[seqPosition][index] = 1;
    }
}

int main()
{
    int **table;
    cudaMalloc((void**) &table, INITIAL_CAPACITY * sizeof(int *));
    for(int i(0); i < INITIAL_CAPACITY; ++i) {
        cudaMalloc((void**) &(table[i]), INITIAL_CAPACITY * sizeof(int));
    }

    while (1) {
        char buffer[100];

        printf("Input a sequence to curl:\n");
        scanf("%s", buffer);

        int i(0);
        int sequence[100];

        for (; buffer[i] != '\0'; ++i) {
            sequence[i] = buffer[i] - '0';
            printf("%d", sequence[i]);
        }
        printf("\n");

        int arraySize = i;

        int *a;
        int iSize = arraySize * sizeof(int);
        cudaMalloc((void**)&a, iSize);
        cudaMemcpy(a, sequence, iSize, cudaMemcpyHostToDevice);

        clock_t start = clock();

        int min = findMin(a, arraySize);

        clock_t stop = clock();
        double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
        printf("Elapsed time: %.3fs\n", elapsed);

        printf("min is %d\n", min);

        cudaFree(a);
    }
    return 0;
}
