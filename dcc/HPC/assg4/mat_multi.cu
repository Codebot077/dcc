#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

int main() {
    int N = 512;
    int size = N * N * sizeof(int);

    int *A, *B, *C;              // Host pointers
    int *dev_A, *dev_B, *dev_C;  // Device pointers

    // Allocate pinned memory on host for faster transfer
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Allocate memory on GPU
    cudaMalloc(&dev_A, size);
    cudaMalloc(&dev_B, size);
    cudaMalloc(&dev_C, size);

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i * N + j;
            B[i * N + j] = j * N + i;
        }
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Start recording

    // Copy data from host to device
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Set block and grid size
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

    // Launch kernel
    matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

    // Copy result back to host
    cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    // Stop event and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print part of the result matrix
    std::cout << "Sample Result (10x10 portion):\n";
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTotal execution time: " << milliseconds / 1000 << " seconds\n";

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

/*
🔧 To compile and run:

1. Save this file as: matmul.cu
2. Compile:           nvcc mat_multi.cu 
3. Run:               ./a.out

Code workflow:
- Initializes two matrices A and B on the host
- Allocates device memory and transfers A and B to the GPU
- Launches a CUDA kernel for matrix multiplication
- Transfers result back to host and prints a sample
- Reports total kernel execution time using CUDA Events
*/
