#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMulKernel(float *a, float *b, float *c, int rowsA, int rowsB, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        float sum = 0;
        for (int i = 0; i < colsA; i++) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

extern "C" void matMul(float *a, float *b, float *c, int rowsA, int rowsB, int colsA, int colsB) {
    float *d_a, *d_b, *d_c;
    int sizeA = rowsA * colsA * sizeof(float);
    int sizeB = rowsB * colsB * sizeof(float);
    int sizeC = rowsA * colsB * sizeof(float);

    cudaMalloc((void **)&d_a, sizeA);
    cudaMalloc((void **)&d_b, sizeB);
    cudaMalloc((void **)&d_c, sizeC);

    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x, (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, rowsA, rowsB, colsA, colsB);

    cudaMemcpy(c, d_c, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}