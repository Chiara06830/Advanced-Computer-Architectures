#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"

using namespace timer;
using namespace timer_cuda;

__global__ void PrefixScanNaive(int* VectorIN, int N) {
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < blockDim.x - 1)
		SMem[threadIdx.x + 1] = VectorIN[GlobalIndex];
	else
		SMem[0] = 0;
	__syncthreads();

	for (int offset = 1; offset <= blockDim.x; offset *= 2) {
		int valueLeft, valueRight;

		if (threadIdx.x >= offset) {
			valueLeft = SMem[threadIdx.x];
			valueRight = SMem[threadIdx.x - offset];
		}
		__syncthreads();

		if (threadIdx.x >= offset)
			SMem[threadIdx.x] = valueLeft + valueRight;

		__syncthreads();
	}
	VectorIN[GlobalIndex] = SMem[threadIdx.x];
}


__global__ void PrefixScanUpDownSweep(int* VectorIN, int N) {
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;

	SMem[threadIdx.x] = VectorIN[GlobalIndex];

	__syncthreads();

	int step = 1;
	for (int limit = blockDim.x / 2; limit > 0; limit /= 2) {

		if (threadIdx.x < limit) {
			int valueRight = (threadIdx.x + 1) * (step * 2) - 1;
			int valueLeft = valueRight - step;
			SMem[valueRight] += SMem[valueLeft];
		}
		step *= 2;
		__syncthreads();
	}

	if (threadIdx.x == 0)
		SMem[blockDim.x - 1] = 0;
	__syncthreads();

	step = blockDim.x / 2;
	for (int limit = 1; limit <= blockDim.x / 2; limit *= 2) {

		if (threadIdx.x < limit) {
			int valueRight = (threadIdx.x + 1) * (step * 2) - 1;
			int valueLeft = valueRight - step;
			int tmp = SMem[valueLeft];
			SMem[valueLeft] = SMem[valueRight];
			SMem[valueRight] += tmp;
		}
		step /= 2;
		__syncthreads();
	}

	VectorIN[GlobalIndex] = SMem[threadIdx.x];
}



void printArray(int* Array, int N, const char str[] = "") {
	std::cout << str;
	for (int i = 0; i < N; ++i)
		std::cout << std::setw(5) << Array[i] << ' ';
	std::cout << std::endl << std::endl;
}


#define DIV(a,b)	(((a) + (b) - 1) / (b))

int main() {
	const int blockDim = 512;
	const int N = blockDim * 131072;


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

	int* devVectorIN1, *devVectorIN2;
	__SAFE_CALL( cudaMalloc(&devVectorIN1, N * sizeof(int)) );
	__SAFE_CALL( cudaMalloc(&devVectorIN2, N * sizeof(int)) );
    	__SAFE_CALL( cudaMemcpy(devVectorIN1, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );
    	__SAFE_CALL( cudaMemcpy(devVectorIN2, VectorIN, N * sizeof(int), cudaMemcpyHostToDevice) );

	int* prefixScan1 = new int[N];
	int* prefixScan2 = new int[N];
	float dev_time1, dev_time2;

	// ------------------- CUDA COMPUTATION 1 ----------------------------------

	dev_TM.start();
	PrefixScanNaive<<<DIV(N, blockDim), blockDim>>>(devVectorIN1, N);
	dev_TM.stop();
	dev_time1 = dev_TM.duration();

	__SAFE_CALL(cudaMemcpy(prefixScan1, devVectorIN1, N * sizeof(int),
                           cudaMemcpyDeviceToHost) );

	// ------------------- CUDA COMPUTATION 2 ----------------------------------

	dev_TM.start();
	PrefixScanUpDownSweep<<<DIV(N, blockDim), blockDim>>>(devVectorIN2, N);
	dev_TM.stop();
	dev_time2 = dev_TM.duration();

	__SAFE_CALL( cudaMemcpy(prefixScan2, devVectorIN2, N * sizeof(int),
                            cudaMemcpyDeviceToHost) );

	// ------------------- CUDA ENDING -----------------------------------------

	std::cout << std::fixed << std::setprecision(1)
              << "KernelTime Naive  : " << dev_time1 << std::endl
			  << "KernelTime Simple : " << dev_time2 << std::endl << std::endl;

	// ------------------- VERIFY ----------------------------------------------

    host_TM.start();

	int* host_result = new int[N];
	std::partial_sum(VectorIN, VectorIN + N, host_result);

    host_TM.stop();

	/*printArray(VectorIN, blockDim, "IN\n");
	printArray(host_result, blockDim, "host\n");
	printArray(prefixScan1, blockDim, "device1\n");
   	 printArray(prefixScan1, blockDim, "device2\n");*/

	if (!std::equal(host_result, host_result + blockDim - 1, prefixScan1 + 1)) {
        	std::cerr << " Error! :  prefixScan1" << std::endl << std::endl;
        	cudaDeviceReset();
        	std::exit(EXIT_FAILURE);
	}
    	if (!std::equal(host_result, host_result + blockDim - 1, prefixScan2 + 1)) {
        	std::cerr << " Error! :  prefixScan 2" << std::endl << std::endl;
        	cudaDeviceReset();
        	std::exit(EXIT_FAILURE);
    }

    // ----------------------- SPEEDUP -----------------------------------------

    float speedup1 = host_TM.duration() / dev_time1;
    float speedup2 = host_TM.duration() / dev_time2;
	std::cout << "Correct result" << std::endl
              << "(1) Speedup achieved: " << speedup1 << " x" << std::endl
              << "(1) Speedup achieved: " << speedup2 << " x"
              << std::endl << std::endl;

    delete[] host_result;
    delete[] prefixScan1;
    delete[] prefixScan2;
    __SAFE_CALL( cudaFree(devVectorIN1) );
    __SAFE_CALL( cudaFree(devVectorIN2) );
    cudaDeviceReset();
}
