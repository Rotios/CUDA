#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define INITIAL_CAPACITY 1024

/******************** Find the min value **************************/
__global__ void minCompare(int *a, int set, bool *check) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tabx = idx * 1024 + set;
    int taby = idy * 1024 + set;

    if (idx == idy) { return; }

    int xval = a[tabx];
    int yval = a[taby];
    
    if (xval == 0 || xval == 1) {
        check[idx] = false;
    } else if (xval > yval) {
        check[idx] = false;
    }
}

__global__ void cudaMin(int *a, int set, bool *check, int* min) {
    int idx = blockIdx.x;

    if (check[idx]) {
        min[0] = a[idx * 1024 + set];
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

/********************** Min and Max Functions ******************************************/
void findMin(int *arr, const int length, const int offset, int *minimum) {
    bool *check;
    int *min;
    int *set;
    int one = 1;

    const int intSize = sizeof(int);
    const int bsize = length * sizeof(bool);

    cudaMalloc((void**)&check, bsize);
    cudaBoolFill<<< dim3(length, 1), 1 >>>(check, length);

    cudaMalloc((void**)&min, intSize);
    //cudaMemcpy(min, (void*)&one, intSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&set, intSize);

    cudaMemcpy(set, (int*)&offset, intSize, cudaMemcpyHostToDevice);

    minCompare<<< dim3(length, length), 1 >>>(arr, *set, check);
    cudaMin<<< dim3(length, 1), 1 >>>(arr, *set, check, min);

    int minhost[1];
    cudaMemcpy(minhost, min, intSize, cudaMemcpyDeviceToHost);

    cudaFree(min);
    cudaFree(check);

    cudaMemcpy(minimum, (void *)&(minhost[0]), intSize, cudaMemcpyHostToDevice);
    //minimum = minhost[0];
}

int findMax(int *arr, const int length) {
    bool *check;
    int *max;

    const int intSize = sizeof(int);
    const int bsize = length * sizeof(bool);

    cudaMalloc((void**)&check, bsize);
    cudaBoolFill<<< dim3(length, 1), 1 >>>(check, length);

    cudaMalloc((void**)&max, intSize);

    maxCompare<<< dim3(length, length), 1 >>>(arr, check);
    cudaMax<<< dim3(length, 1), 1 >>>(arr, check, max);

    int maxhost[1];
    cudaMemcpy(maxhost, max, intSize, cudaMemcpyDeviceToHost);

    cudaFree(max);
    cudaFree(check);

    return maxhost[0];
}

/********************* Find the Curl *****************************************/
int findCurl(int *sequence, int *table, int length){
    int *tempResults;
    cudaMalloc((void **) &tempResults, (length >> 1) * sizeof(int));

    for(int i(0); i < (length >> 1); ++i) {
        //int *p = &(table[i][(length - 1) - i]);
        //findMin(p, length, &(tempResults[i]));
        findMin(table, i+1, (length - 1) - i, &(tempResults[i]));
    }
    int curl = findMax(tempResults, length);

    cudaFree(tempResults);

    return curl;
}

void printTable(int *table, int length) {
    int *CPUTable;
    CPUTable = (int *) malloc(INITIAL_CAPACITY * INITIAL_CAPACITY * sizeof(int));
    cudaMemcpy(CPUTable, table, INITIAL_CAPACITY * INITIAL_CAPACITY * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i(0); i < length; ++i) {
        for(int j(0); j < length; ++j) {
            printf("%d ", CPUTable[(i * 1024) + j]);
        }
        printf("\n");
    }

    free(CPUTable);
}

__global__ void fillColumn(int *sequence, int *table, int *seqPosition) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int index = *seqPosition;
    int value = 1;
    
    if(row == index){}
    else if(sequence[index - (row + 1)] == sequence[index]) {
        value = table[(row * 1024) + (index - (row + 1))] + 1;
    }

    table[(row * 1024) + index] = value;
}

void initializeTable(int *sequence, int *table, int length) {

    int *index;
    cudaMalloc((void **)&index, sizeof(int *));

    for(int i(0); i < length; ++i) {
        cudaMemcpy(index, (void *)&i, sizeof(int), cudaMemcpyHostToDevice);
        fillColumn <<< dim3(i + 1, 1), 1 >>>(sequence, table, index);
        //printTable(table, i);
    }

    cudaFree(index);
}

int main()
{
    int *table;

    cudaMalloc((void**)&table, (INITIAL_CAPACITY * INITIAL_CAPACITY) * sizeof(int));

    while (1) {

        cudaMemset(table, 0, (INITIAL_CAPACITY * INITIAL_CAPACITY) * sizeof(int));
        
        char buffer[100];

        printf("Input a sequence to curl:\n");
        scanf("%s", buffer);

        int i(0);
        int sequence[100];

        for (; buffer[i] != '\0'; ++i) {
            sequence[i] = buffer[i] - '0';
        }

        int arraySize = i;

        int *a;
        int iSize = arraySize * sizeof(int);
        cudaMalloc((void**)&a, iSize);
        cudaMemcpy(a, sequence, iSize, cudaMemcpyHostToDevice);

        printTable(table, arraySize);

        initializeTable(a, table, arraySize);

        printTable(table, arraySize);

        clock_t start = clock();

        int curl = findCurl(a, table, arraySize);

        clock_t stop = clock();
        double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
        printf("Elapsed time: %.3fs\n", elapsed);

        printf("curl is %d\n", curl);

        cudaFree(a);
    }
    return 0;
}
