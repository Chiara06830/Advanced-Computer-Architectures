#include <iostream>
#include <chrono>
#include <random>
#include "Timer.cuh"
#include "CheckError.cuh"

using namespace timer;

// Macros
#define DIV(a, b)   (((a) + (b) - 1) / (b))

const int N  = 16777216;
#define BLOCK_SIZE 256

// v1: with shared mem and divergent
__global__ void ReduceKernel_shared_div(int* VectorIN, int N) {
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

// v2: with shared mem and less divergent
__global__ void ReduceKernel_shared(int* VectorIN, int N) {
	__shared__ int SMem[1024];
	int GlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	SMem[threadIdx.x] = VectorIN[GlobalIndex];
	__syncthreads();
	for (int i = 1; i < blockDim.x; i *= 2) {
		int index = threadIdx.x * i * 2;
		if (index < blockDim.x)
			SMem[index] += SMem[index + i];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		VectorIN[blockIdx.x] = SMem[0];
}

int main() {
    
	// ------------------- INIT ------------------------------------------------

	// Random Engine Initialization
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::uniform_int_distribution<int> distribution(1, 100);

	Timer<HOST> host_TM;
	Timer<DEVICE> dev_TM;

	// ------------------ HOST INIT --------------------------------------------

	int* VectorIN = new int[N];
	for (int i = 0; i < N; ++i)
		VectorIN[i] = distribution(generator);

	// ------------------- CUDA INIT -------------------------------------------
	// v3: global with task parallelism for data transfer/kernel computation overlapping
	//cudaStream_t stream0, stream1;
	//cudaStreamCreate( &stream0);
	//cudaStreamCreate( &stream1);
	//int *devVectorIN0; // device memory for stream 0
	//int *devVectorIN1; // device memory for stream 1

	int* devVectorIN;
	SAFE_CALL( cudaMalloc(&devVectorIN0, N * sizeof(int)) );
	SAFE_CALL( cudaMemcpy(devVectorIN0, VectorIN, N * sizeof(int),
	            cudaMemcpyHostToDevice) );
	SAFE_CALL( cudaMalloc(&devVectorIN1, N * sizeof(int)) );
	SAFE_CALL( cudaMemcpy(devVectorIN1, VectorIN, N * sizeof(int),
	            cudaMemcpyHostToDevice) );
	float dev_time; 
	
	// ------------------- CUDA COMPUTATION ----------------------------------
	int sum;

	std::cout<<"Starting computation on DEVICE "<<std::endl;

	dev_TM.start();
	
	ReduceKernel_shared<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE>>>
        	       (devVectorIN, N);
	ReduceKernel_shared<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE>>>
                (devVectorIN, DIV(N, BLOCK_SIZE));
	ReduceKernel_shared<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>
             (devVectorIN, DIV(N, BLOCK_SIZE * BLOCK_SIZE));
        
        //for (int i=0; i<N; i+=BLOCK_SIZE*2) {
		//cudaMemcpyAsync(devVectorIN0, VectorIN+i, N*sizeof(float), cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0);
		//cudaMemcpyAsync(devVectorIN0, VectorIN+i+BLOCK_SIZE, N*sizeof(float), stream1);
		
		//ReduceKernel_shared<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE, stream0>>>
               //       (devVectorIN0, N);
		//ReduceKernel_shared<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE, stream0>>>
               //        (devVectorIN0, DIV(N, BLOCK_SIZE));
		//ReduceKernel_shared<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE, stream0>>>
                //       (devVectorIN0, DIV(N, BLOCK_SIZE * BLOCK_SIZE));
                       
                //ReduceKernel_shared<<<DIV(N, BLOCK_SIZE), BLOCK_SIZE, stream1>>>
                //       (devVectorIN1, N);
		//ReduceKernel_shared<<<DIV(N, BLOCK_SIZE* BLOCK_SIZE), BLOCK_SIZE, stream1>>>
                //       (devVectorIN1, DIV(N, BLOCK_SIZE));
		//ReduceKernel_shared<<<DIV(N, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE, stream1>>>
                //       (devVectorIN1, DIV(N, BLOCK_SIZE * BLOCK_SIZE));
                       
		//cudaMemcpyAsync(devVectorIN0, VectorIN+i, N*sizeof(float),stream0);
		//cudaMemcpyAsync(devVectorIN0, VectorIN+i+BLOCK_SIZE, N*sizeof(float), stream1);
	//}

	dev_TM.stop();
	dev_time = dev_TM.duration();
	CHECK_CUDA_ERROR;

	std::cout << std::setprecision(3)
              << "DeviceTime            : " << dev_TM.duration() << std::endl;

	SAFE_CALL( cudaMemcpy(&sum, devVectorIN, sizeof(int),
                            cudaMemcpyDeviceToHost) );

	// ------------------- HOST ------------------------------------------------
	   
	host_TM.start();

	// v0: sequential
	int host_sum = std::accumulate(VectorIN, VectorIN + N, 0);

	host_TM.stop();

	std::cout << std::setprecision(3)
              << "HostTime            : " << host_TM.duration() << std::endl
              << std::endl;

    // ------------------------ VERIFY -----------------------------------------

    if (host_sum != sum) {
        std::cerr << std::endl
                  << "Error! Wrong result. Host value: " << host_sum
                  << " , Device value: " << sum
                  << std::endl << std::endl;
        cudaDeviceReset();
        std::exit(EXIT_FAILURE);
    }

    //-------------------------- SPEEDUP ---------------------------------------

    float speedup = host_TM.duration() / dev_time;

    std::cout << "Correct result" << std::endl
              << "Speedup achieved: " << std::setprecision(3)
              << speedup << " x" << std::endl << std::endl;

    std::cout << host_TM.duration() << ";" << dev_TM.duration() << ";" << host_TM.duration() / dev_TM.duration() << std::endl;

    delete[] VectorIN;
    SAFE_CALL( cudaFree(devVectorIN) );
    cudaDeviceReset();
}
