#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"

using namespace timer;
//using namespace timer_cuda;


const int N  = 100;
#define BLOCK_SIZE 32


__global__ void ReduceKernelDivergent(int* VectorIN, int N) {
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	SMem[threadIdx.x] = VectorIN[GlobalIndex];
	__syncthreads();

	for (int i = 1; i < blockDim.x; i *= 2) {
		if (threadIdx.x % (i * 2) == 0)
			SMem[threadIdx.x] += SMem[threadIdx.x + i];

		__syncthreads();
	}
	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = SMem[0];
}


__global__ void ReduceKernel(int* VectorIN, int N) {
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	SMem[threadIdx.x] = VectorIN[GlobalIndex];
	__syncthreads();

	for (int i = 1; i < blockDim.x; i *= 2) {
		int index = threadIdx.x * i * 2;
		if (index < blockDim.x)
		//if (threadIdx.x < blockDim.x / (i * 2))
			SMem[index] += SMem[index + i];

		__syncthreads();
	}

	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = SMem[0];
}



//last warp optimization
__global__ void ReduceKernel2(int* VectorIN, int N) {
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	SMem[threadIdx.x] = VectorIN[GlobalIndex];
	__syncthreads();

	for (int i = 1; i < blockDim.x / 32; i *= 2) {
		int index = threadIdx.x * i * 2;
		if (index < blockDim.x)
			SMem[index] += SMem[index + i];

		__syncthreads();
	}
	if (threadIdx.x < 32) {
		for (int i = blockDim.x / 32; i < blockDim.x; i *= 2) {
			int index = threadIdx.x * i * 2;
			if (index < blockDim.x)
				SMem[index] += SMem[index + i];
		}
	}

	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = SMem[0];
}

#define DIV(a, b)   (((a) + (b) - 1) / (b))

int main() {
    
    // ------------------- INIT ------------------------------------------------

    // Random Engine Initialization
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    timer::Timer<HOST> host_TM;
    timer_cuda::Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------

	int* devVectorIN1, *devVectorIN2, *devVectorIN3;
	__SAFE_CALL( cudaMalloc(&devVectorIN1, N * sizeof(int)) );
	__SAFE_CALL( cudaMalloc(&devVectorIN2, N * sizeof(int)) );
	__SAFE_CALL( cudaMalloc(&devVectorIN3, N * sizeof(int)) );
	__SAFE_CALL( cudaMemcpy(devVectorIN1, VectorIN, N * sizeof(int),
                 cudaMemcpyHostToDevice) );
	__SAFE_CALL( cudaMemcpy(devVectorIN2, VectorIN, N * sizeof(int),
                 cudaMemcpyHostToDevice) );
	__SAFE_CALL( cudaMemcpy(devVectorIN3, VectorIN, N * sizeof(int),
                 cudaMemcpyHostToDevice) );

	int sum1, sum2, sum3;
	float dev_time1, dev_time2, dev_time3;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

	
    std::cout<<"Starting V1 (divergent kernel) computation on DEVICE "<<std::endl;

    dev_TM.start();
	ReduceKernelDivergent<<<DIV(N, blockDim), blockDim>>>
                        (devVectorIN1, N);
	ReduceKernelDivergent<<<DIV(N, blockDim * blockDim), blockDim>>>
                         (devVectorIN1, DIV(N, blockDim));
	ReduceKernelDivergent<<<DIV(N, blockDim * blockDim * blockDim), blockDim>>>
                         (devVectorIN1, DIV(N, blockDim * blockDim));

	dev_TM.stop();
	dev_time1 = dev_TM.duration();
	__CUDA_ERROR("Reduce v1 on GPU");

	__SAFE_CALL( cudaMemcpy(&sum1, devVectorIN1, sizeof(int),
                            cudaMemcpyDeviceToHost) );

	// ------------------- CUDA COMPUTATION 2 ----------------------------------
	
    std::cout<<"Starting V2 computation on DEVICE "<<std::endl;
    dev_TM.start();

	ReduceKernel<<<DIV(N, blockDim), blockDim>>>
                       (devVectorIN2, N);
	ReduceKernel<<<DIV(N, blockDim * blockDim), blockDim>>>
                       (devVectorIN2, DIV(N, blockDim));
	ReduceKernel<<<DIV(N, blockDim * blockDim * blockDim), blockDim>>>
                       (devVectorIN2, DIV(N, blockDim * blockDim));

	dev_TM.stop();
	dev_time2 = dev_TM.duration();
	__CUDA_ERROR("Reduce v2");

	__SAFE_CALL( cudaMemcpy(&sum2, devVectorIN2, sizeof(int),
                            cudaMemcpyDeviceToHost) );

	// ------------------- CUDA COMPUTATION 3 ----------------------------------

    std::cout<<"Starting V3 (a more optimized version) computation on DEVICE "<<std::endl;
	dev_TM.start();

	ReduceKernel2<<<DIV(N, blockDim), blockDim>>>
                    (devVectorIN3, N);
	ReduceKernel2<<<DIV(N, blockDim * blockDim), blockDim>>>
                    (devVectorIN3, N / blockDim);
	ReduceKernel2<<<DIV(N, blockDim * blockDim * blockDim), blockDim>>>
                    (devVectorIN3, N / (blockDim * blockDim));

	dev_TM.stop();
	dev_time3 = dev_TM.duration();
	__CUDA_ERROR("Reduce v3");

	__SAFE_CALL( cudaMemcpy(&sum3, devVectorIN3, sizeof(int),
                            cudaMemcpyDeviceToHost) );

	// ------------------- HOST ------------------------------------------------
    host_TM.start();

	int host_sum = std::accumulate(VectorIN, VectorIN + N, 0);

    host_TM.stop();

    std::cout << std::setprecision(3)
              << "KernelTime Divergent: " << dev_time1 << std::endl
              << "KernelTime Simple   : " << dev_time2 << std::endl
              << "KernelTime Opt      : " << dev_time3 << std::endl
              << "HostTime            : " << host_TM.duration() << std::endl
              << std::endl;

    // ------------------------ VERIFY -----------------------------------------

    if (host_sum != sum1) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum1
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
    if (host_sum != sum2) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum2
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }
    if (host_sum != sum3) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum3
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }

    //-------------------------- SPEEDUP ---------------------------------------

    float speedup1 = host_TM.duration() / dev_time1;
    float speedup2 = host_TM.duration() / dev_time2;
    float speedup3 = host_TM.duration() / dev_time3;

    std::cout << "Correct result" << std::endl
              << "(v1) Speedup achieved: " << std::setprecision(3)
              << speedup1 << " x" << std::endl
              << "(v2) Speedup achieved: "
              << speedup2 << " x" << std::endl
              << "(v3) Speedup achieved: "
              << speedup3 << " x" << std::endl << std::endl;

    delete[] VectorIN;
    __SAFE_CALL( cudaFree(devVectorIN1) );
    __SAFE_CALL( cudaFree(devVectorIN2) );
    __SAFE_CALL( cudaFree(devVectorIN3) );
    cudaDeviceReset();
}
