#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define INITIAL_CAPACITY 1024

/******************** Find the min value **************************/
__global__ void minCompare(int *a, int *set, bool *check, int *capacity) {
    int cap = capacity[0];
    int offset = set[0];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int tabx = idx + cap + offset;
    int taby = idy + cap + offset;

    if (idx == idy) { return; }

    int xval = a[tabx];
    int yval = a[taby];
    
    if(yval <= 1) {}
    else if (xval <= 1) {
        check[idx] = false;
    } else if (xval > yval) {
        check[idx] = false;
    }
}

__global__ void cudaMin(int *a, int *set, bool *check, int* min, int *capacity) {
    int idx = blockIdx.x;

    if (check[idx]) {
        min[0] = a[idx + capacity[0] + set[0]];
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
void findMin(int *arr, const int length, const int offset, int *minimum, int *capacity) {
    //length - 1 = row, offset = location of first element

    bool *check;
    int *min;
    int *set;
    int *row = (int*) malloc(sizeof(int));
    int one = 1;
    const int intSize = sizeof(int);
    const int bsize = length * sizeof(bool);

    cudaMalloc((void**)&check, bsize);
    cudaBoolFill<<< dim3(length, 1), 1 >>>(check, length);

    cudaMalloc((void**)&min, intSize);
    cudaMemcpy(min, (void*)&one, intSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&set, intSize);
    cudaMemcpy(set, (int*)&offset, intSize, cudaMemcpyHostToDevice);

    cudaMemcpy(row, capacity, intSize, cudaMemcpyDeviceToHost);
    row[0] = row[0] * (length - 1);
   
    printf("offset = %d    length = %d     row = %d\n", offset, length, row[0]);

    int *row2;
    cudaMalloc((void**) &row2, intSize);
    cudaMemcpy(row2, row, intSize, cudaMemcpyHostToDevice);

    minCompare<<< dim3(length, length), 1 >>>(arr, set, check, row2);
    cudaMin<<< dim3(length, 1), 1 >>>(arr, set, check, min, row2);

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
int findCurl(int *sequence, int *table, int length, int capacity){
    int *tempResults;
    cudaMalloc((void **) &tempResults, (length >> 1) * sizeof(int));
    int *cap;
    cudaMalloc((void **) &cap, sizeof(int));
    cudaMemcpy(cap, (int*)&capacity, sizeof(int), cudaMemcpyHostToDevice);

    for(int i(0); i < (length >> 1); ++i) {
        //int *p = &(table[i][(length - 1) - i]);
        //findMin(p, length, &(tempResults[i]));
        findMin(table, i+1, (length - 1) - i, &(tempResults[i]), cap);
    }

    int *results = (int *) malloc((length >> 1) * sizeof(int));
    cudaMemcpy(results, tempResults, (length >> 1) * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i(0); i < (length >> 1); ++i) {
        printf("%d ", results[i]);
    }
    printf("\n");

    int curl = findMax(tempResults, length);

    cudaFree(tempResults);

    return curl;
}

void printTable(int *table, int length, int capacity) {
    int *CPUTable;
    CPUTable = (int *) malloc(capacity * capacity * sizeof(int));
    cudaMemcpy(CPUTable, table, capacity * capacity * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i(0); i < length; ++i) {
        for(int j(0); j < length; ++j) {
            printf("%d ", CPUTable[(i * capacity) + j]);
        }
        printf("\n");
    }

    free(CPUTable);
}

__global__ void fillColumn(int *sequence, int *table, int *seqPosition, int *cap) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int index = *seqPosition;
    int capacity = *cap;
    int value = 1;
    
    if(row == index){}
    else if(sequence[index - (row + 1)] == sequence[index]) {
        int t = table[(row * capacity) + (index - (row + 1))];
        if(t == 0) {
            value = 2;
        } else {
            value = table[(row * capacity) + (index - (row + 1))] + 1;
        }
    }

    table[(row * capacity) + index] = value;
}

void initializeTable(int *sequence, int *table, int length, int capacity) {

    int *index;
    cudaMalloc((void **)&index, sizeof(int));
    int *cap;
    cudaMalloc((void **)&cap, sizeof(int));
    cudaMemcpy(cap, (void *)&capacity, sizeof(int), cudaMemcpyHostToDevice);

    for(int i(0); i < length; ++i) {
        cudaMemcpy(index, (void *)&i, sizeof(int), cudaMemcpyHostToDevice);
        fillColumn<<< dim3(i + 1, 1), 1 >>>(sequence, table, index, cap);
    }

    cudaFree(index);
}

int main() {
    int *table;
    int capacity = INITIAL_CAPACITY;

    cudaMalloc((void**)&table, (INITIAL_CAPACITY * INITIAL_CAPACITY) * sizeof(int));

    while (1) {

        cudaMemset(table, 0, (capacity * capacity) * sizeof(int));
        
        char buffer[100];
        printf("Input a sequence to curl:\n");
        scanf("%s", buffer);

        int i(0);
        int sequence[INITIAL_CAPACITY];
        for (; buffer[i] != '\0'; ++i) {
            sequence[i] = buffer[i] - '0';
        }

        int arraySize = i;
        int *a;
        int iSize = arraySize * sizeof(int);
        cudaMalloc((void**)&a, iSize);
        cudaMemcpy(a, sequence, iSize, cudaMemcpyHostToDevice);

        initializeTable(a, table, arraySize, capacity);
        printTable(table, arraySize, capacity);

        clock_t start = clock();

        int *size;
        cudaMalloc((void**)&size, sizeof(int));
        int *cap;
        cudaMalloc((void **)&cap, sizeof(int));
        cudaMemcpy(cap, (void *)&capacity, sizeof(int), cudaMemcpyHostToDevice);
        int curl = 0;

        while(curl != 1) {
            curl = findCurl(a, table, arraySize, capacity);
            sequence[arraySize] = curl;
            cudaMemcpy(size, (int*)&arraySize, sizeof(int), cudaMemcpyHostToDevice);
            iSize = ++arraySize * sizeof(int);
            cudaMalloc((void**)&a, iSize);
            cudaMemcpy(a, sequence, iSize, cudaMemcpyHostToDevice);
            fillColumn<<< dim3(arraySize, 1), 1 >>>(a, table, size, cap);
        }

        clock_t stop = clock();
        double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
        printf("Elapsed time: %.3fs\n", elapsed);
        printf("curl is %d\n", curl);

        cudaFree(a);
    }
    return 0;
}
