#include <chrono>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__
void matrixMultiplicationKernel(const int* d_matrixA,
                                const int* d_matrixB,
                                int        N,
                                int*       d_matrixC) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < N && row < N) {
		int sum = 0;
		for (int k = 0; k < N; k++)
		     sum += d_matrixA[row * N + k] * d_matrixB[k * N + col];
		d_matrixC[row * N + col] = sum;
	}
    
}

const int N = 3000;

int main() {
    Timer<DEVICE> TM_device;
    Timer<HOST>   TM_host;
    // -------------------------------------------------------------------------
    // HOST MEMORY ALLOCATION
    int* h_matrixA    = new int[N * N];
    int* h_matrixB    = new int[N * N];
    int* h_matrix_tmp = new int[N * N]; // <-- used for device result
    int* h_matrixC    = new int[N * N];

    // -------------------------------------------------------------------------
    // HOST INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N * N; i++) {
        h_matrixA[i] = distribution(generator);
        h_matrixB[i] = distribution(generator);
    }
    // -------------------------------------------------------------------------
    // HOST EXECUTIION
    TM_host.start();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                 sum += h_matrixA[i * N + k] * h_matrixB[k * N + j];
            h_matrixC[i * N + j] = sum;
        }
    }

    TM_host.stop();
    TM_host.print("MatrixMultiplication host:   ");

    // -------------------------------------------------------------------------
    // DEVICE MEMORY ALLOCATION
    int *d_matrixA, *d_matrixB, *d_matrixC;
    SAFE_CALL( cudaMalloc( &d_matrixA, N * N * sizeof(int) ));
	SAFE_CALL( cudaMalloc( &d_matrixB, N * N * sizeof(int) ));
    SAFE_CALL( cudaMalloc( &d_matrixC, N * N * sizeof(int) ));

    // -------------------------------------------------------------------------
    // COPY DATA FROM HOST TO DEVIE
    SAFE_CALL( cudaMemcpy( d_matrixA, h_matrixA, N * N * sizeof(int), cudaMemcpyHostToDevice));
	SAFE_CALL( cudaMemcpy( d_matrixB, h_matrixB, N * N * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // DEVICE EXECUTION
    TM_device.start();

    dim3 DimGrid(N/BLOCK_SIZE_X,N/BLOCK_SIZE_Y, 1);
	if (N%BLOCK_SIZE_X) DimGrid.x++;
	if (N%BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    matrixMultiplicationKernel<<<DimGrid, DimBlock>>>
    (d_matrixA, d_matrixB, N, d_matrixC);

    TM_device.stop();
    CHECK_CUDA_ERROR
    TM_device.print("MatrixMultiplication device: ");

    std::cout << std::setprecision(1)
              << "Speedup: " << TM_host.duration() / TM_device.duration()
              << "x\n\n";

    // -------------------------------------------------------------------------
    // COPY DATA FROM DEVICE TO HOST
    SAFE_CALL( cudaMemcpy(h_matrix_tmp, d_matrixC, N * N * sizeof(int), cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // RESULT CHECK
    for (int i = 0; i < N * N; i++) {
        if (h_matrixC[i] != h_matrix_tmp[i]) {
            std::cerr << "wrong result at: ("
                      << (i / N) << ", " << (i % N) << ")"
                      << "\nhost:   " << h_matrixC[i]
                      << "\ndevice: " << h_matrix_tmp[i] << "\n\n";
            cudaDeviceReset();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "<> Correct\n\n";

    // -------------------------------------------------------------------------
    // HOST MEMORY DEALLOCATION
    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_matrixC;
    delete[] h_matrix_tmp;

    // -------------------------------------------------------------------------
    // DEVICE MEMORY DEALLOCATION
    SAFE_CALL( cudaFree( d_matrixA ) );
    SAFE_CALL( cudaFree( d_matrixB ) );
    SAFE_CALL( cudaFree( d_matrixC ) );

    // -------------------------------------------------------------------------
    cudaDeviceReset();
}
