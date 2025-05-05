#include <iostream>  
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to add two vectors
__global__ void addVectors(int* A, int* B, int* C, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() 
{
    int n = 1000000; // Size of vectors
    int size = n * sizeof(int);

    int *A, *B, *C; // Host pointers
    int *dev_A, *dev_B, *dev_C; // Device pointers

    // Allocate pinned memory on the host for faster transfers
    cudaMallocHost(&A, size);  
    cudaMallocHost(&B, size);  
    cudaMallocHost(&C, size);

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&dev_A, size);  
    cudaMalloc(&dev_B, size);  
    cudaMalloc(&dev_C, size);

    // Create CUDA events to time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Start timing

    // Copy data from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);  
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Define execution configuration
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    addVectors<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C, n);

    // Copy the result back to the host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Stop timing and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print a few sample results
    cout << "Sample output (first 10 elements): ";
    for (int i = 0; i < 10; i++) {
        cout << C[i] << " ";
    }
    cout << endl;

    // Print total execution time in seconds
    cout << "Total kernel execution time: " << milliseconds / 1000 << " seconds" << endl;

    // Free device and host memory
    cudaFree(dev_A);  
    cudaFree(dev_B);  
    cudaFree(dev_C);  
    cudaFreeHost(A);  
    cudaFreeHost(B);  
    cudaFreeHost(C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

/*
To compile and run:

1. Compile: nvcc vect_add.cu 
2. Run:     ./a.out
*/