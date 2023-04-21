#include "LeniaEnvironment.cuh"
#include "KernelFunctions.cuh"

#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <fstream>

namespace {
	constexpr int kThreadsPerBlock = 8;
}

LeniaEnvironment::LeniaEnvironment(std::string file, int fieldSize, int kernelRadius, int stepsPerTimeUnit, float mu, float sigma,
	KernelType kernelType, GrowthType growthType) :
	fieldSize(fieldSize), kernelSize(2 * kernelRadius + 1), padSize(kernelRadius),
	paddedFieldSize(fieldSize + 2 * padSize),
	volume(fieldSize * fieldSize * fieldSize),
	volumePadded(paddedFieldSize * paddedFieldSize * paddedFieldSize),
	volumeKernel(kernelSize * kernelSize * kernelSize),
	mu(mu), sigma(sigma), 
	stepsPerTimeUnit(stepsPerTimeUnit), timeDelta(1.0 / stepsPerTimeUnit) {

	// malloc arrays
	imgCpu = (float*)calloc(volume, sizeof(*imgCpu));
	res = cudaMalloc((void**)(&imgGpu), volumePadded * sizeof(*imgGpu));
	assert(res == cudaSuccess);
	res = cudaMalloc((void**)(&growth), volume * sizeof(*growth));
	assert(res == cudaSuccess);
	res = cudaMalloc((void**)(&kernel), volumeKernel * sizeof(*kernel));
	assert(res == cudaSuccess);

	// set kernel launch params
	int gridSize = ceil(float(fieldSize) / float(kThreadsPerBlock));
	gridDim = dim3(gridSize, gridSize, gridSize);
	blockDim = dim3(kThreadsPerBlock, kThreadsPerBlock, kThreadsPerBlock);

	// initialize field
	std::ifstream input(file, std::ios::binary);
	input.read((char *)imgCpu, volume * sizeof(*imgCpu));

	// move field to gpu
	cudaMemcpy3DParms cpuToGpu{ 0 };
	cpuToGpu.srcPtr = make_cudaPitchedPtr(imgCpu,
		sizeof(*imgCpu) * fieldSize, fieldSize, fieldSize);
	cpuToGpu.dstPtr = make_cudaPitchedPtr(imgGpu,
		sizeof(*imgGpu) * paddedFieldSize, paddedFieldSize, paddedFieldSize);
	cpuToGpu.extent = make_cudaExtent(fieldSize * sizeof(*imgCpu), fieldSize, 
		fieldSize);
	cpuToGpu.srcPos = make_cudaPos(0 * sizeof(*imgCpu), 0, 0);
	cpuToGpu.dstPos = make_cudaPos(padSize * sizeof(*imgGpu), padSize, padSize);
	cpuToGpu.kind = cudaMemcpyHostToDevice;
	res = cudaMemcpy3D(&cpuToGpu);
	assert(res == cudaSuccess);

	// initialize convolution kernel

	// power of 2 nearest to volumeKernel and smaller than it
	int sumReductionBufSize = volumeKernel;
	sumReductionBufSize--;
	sumReductionBufSize |= sumReductionBufSize >> 1;
	sumReductionBufSize |= sumReductionBufSize >> 2;
	sumReductionBufSize |= sumReductionBufSize >> 4;
	sumReductionBufSize |= sumReductionBufSize >> 8;
	sumReductionBufSize |= sumReductionBufSize >> 16;
	sumReductionBufSize++;
	sumReductionBufSize >>= 1;

	// malloc array for sum reduction
	float* sumReductionBuf;
	res = cudaMalloc((void**)(&sumReductionBuf), sumReductionBufSize * sizeof(*sumReductionBuf));
	assert(res == cudaSuccess);

	// calculate kernel
	switch (kernelType) {
	case KernelType::kUniform:
		setUniformKernel <<< gridDim, blockDim >>> (kernel, kernelSize, sumReductionBuf, sumReductionBufSize);
		break;
	case KernelType::kMove:
		setMoveKernel <<< gridDim, blockDim >>> (kernel, kernelSize, sumReductionBuf, sumReductionBufSize);
		break;
	case KernelType::kExponential:
		setExponentialKernel <<< gridDim, blockDim >>> (kernel, kernelSize, sumReductionBuf, sumReductionBufSize);
		break;
	case KernelType::kPolynomial:
		setPolynomialKernel <<< gridDim, blockDim >>> (kernel, kernelSize, sumReductionBuf, sumReductionBufSize);
		break;
	case KernelType::kRectangular:
		setRectangularKernel <<< gridDim, blockDim >>> (kernel, kernelSize, sumReductionBuf, sumReductionBufSize);
		break;
	case KernelType::kGol:
		setGolKernel <<< gridDim, blockDim >>> (kernel, kernelSize, sumReductionBuf, sumReductionBufSize);
		break;
	}

	// free array for sum reduction
	res = cudaFree(sumReductionBuf);
	assert(res == cudaSuccess);

	// set growth type function
	switch (growthType) {
	case GrowthType::kDouble:
		applyGrowth = applyDoubleGrowth;
		break;
	case GrowthType::kExponential:
		applyGrowth = applyExponentialGrowth;
		break;
	case GrowthType::kPolynomial:
		applyGrowth = applyPolynomialGrowth;
		break;
	case GrowthType::kRectangular:
		applyGrowth = applyRectangularGrowth;
		break;
	}
	res = cudaGetLastError();
	assert(res == cudaSuccess);

	// initialize memcpy params for padding
	params = MemcpyParams(fieldSize, paddedFieldSize, padSize, imgGpu);
};

void LeniaEnvironment::Convolute() {
	convolute <<< gridDim, blockDim >>> (
		imgGpu, kernel, growth, paddedFieldSize, kernelSize, fieldSize);
}

void LeniaEnvironment::SetPadding() {
	res = cudaMemcpy3D(&(params.rowLo));
	assert(res == cudaSuccess);
	res = cudaMemcpy3D(&(params.rowHi));
	assert(res == cudaSuccess);
	res = cudaMemcpy3D(&(params.colLo));
	assert(res == cudaSuccess);
	res = cudaMemcpy3D(&(params.colHi));
	assert(res == cudaSuccess);
	res = cudaMemcpy3D(&(params.layLo));
	assert(res == cudaSuccess);
	res = cudaMemcpy3D(&(params.layHi));
	assert(res == cudaSuccess);
}

void LeniaEnvironment::ApplyGrowth() {
	(*applyGrowth) <<< gridDim, blockDim >>> (
		growth, fieldSize, mu, sigma);
}

void LeniaEnvironment::SumAndCut() {
	sumAndCut <<< gridDim, blockDim >>> (
		imgGpu, growth, fieldSize, padSize, paddedFieldSize, timeDelta);
}

void LeniaEnvironment::Step() {
	SetPadding();
	Convolute();
	ApplyGrowth();
	SumAndCut();
}

float* LeniaEnvironment::GetResult() {
	cudaMemcpy3DParms gpuToCpu{ 0 };
	gpuToCpu.dstPtr = make_cudaPitchedPtr(imgCpu,
		sizeof(*imgCpu) * fieldSize, fieldSize, fieldSize);
	gpuToCpu.srcPtr = make_cudaPitchedPtr(imgGpu,
		sizeof(*imgGpu) * paddedFieldSize, paddedFieldSize, paddedFieldSize);
	gpuToCpu.extent = make_cudaExtent(fieldSize * sizeof(*imgCpu), fieldSize,
		fieldSize);
	gpuToCpu.dstPos = make_cudaPos(0 * sizeof(*imgCpu), 0, 0);
	gpuToCpu.srcPos = make_cudaPos(padSize * sizeof(*imgCpu), padSize, padSize);
	gpuToCpu.kind = cudaMemcpyDeviceToHost;
	res = cudaMemcpy3D(&gpuToCpu);
	assert(res == cudaSuccess);

	return imgCpu;
}

float* LeniaEnvironment::StepTimeUnit() {
	for (int i = 0; i < stepsPerTimeUnit; i++) {
		Step();
	}
	return GetResult();
}

void LeniaEnvironment::PutStepTimeUnit(std::string file) {
	float* result = StepTimeUnit();
	std::ofstream output(file, std::ios::binary);
	output.write((char *)result, volume * sizeof(*result));
}

void LeniaEnvironment::PutResult(std::string file) {
	float* result = GetResult();
	std::ofstream output(file, std::ios::binary);
	output.write((char*)result, volume * sizeof(*result));
}


LeniaEnvironment::~LeniaEnvironment() {
	free(imgCpu);
	res = cudaFree(imgGpu);
	assert(res == cudaSuccess);
	res = cudaFree(growth);
	assert(res == cudaSuccess);
	res = cudaFree(kernel);
	assert(res == cudaSuccess);
};

LeniaEnvironment::MemcpyParams::MemcpyParams(int fieldSize, int paddedFieldSize, int padSize, float* ptr) :
	rowLo{ 0 }, rowHi{ 0 }, colLo{ 0 }, colHi{ 0 }, layLo{ 0 }, layHi{ 0 } {

	cudaPitchedPtr pitchedPtr = make_cudaPitchedPtr(ptr,
		sizeof(float) * paddedFieldSize, paddedFieldSize, paddedFieldSize);
	rowLo.srcPtr = pitchedPtr;
	rowHi.srcPtr = pitchedPtr;
	colLo.srcPtr = pitchedPtr;
	colHi.srcPtr = pitchedPtr;
	layLo.srcPtr = pitchedPtr;
	layHi.srcPtr = pitchedPtr;
	rowLo.dstPtr = pitchedPtr;
	rowHi.dstPtr = pitchedPtr;
	colLo.dstPtr = pitchedPtr;
	colHi.dstPtr = pitchedPtr;
	layLo.dstPtr = pitchedPtr;
	layHi.dstPtr = pitchedPtr;

	cudaExtent rowExtent = make_cudaExtent(padSize * sizeof(float), fieldSize,
		fieldSize);
	cudaExtent colExtent = make_cudaExtent(paddedFieldSize * sizeof(float), padSize,
		fieldSize);
	cudaExtent layExtent = make_cudaExtent(paddedFieldSize * sizeof(float), paddedFieldSize,
		padSize);
	rowLo.extent = rowExtent;
	rowHi.extent = rowExtent;
	colLo.extent = colExtent;
	colHi.extent = colExtent;
	layLo.extent = layExtent;
	layHi.extent = layExtent;

	rowLo.srcPos = make_cudaPos((paddedFieldSize - 2 * padSize) * sizeof(float), padSize, padSize);
	rowLo.dstPos = make_cudaPos(0 * sizeof(float), padSize, padSize);
	rowHi.srcPos = make_cudaPos(padSize * sizeof(float), padSize, padSize);
	rowHi.dstPos = make_cudaPos((paddedFieldSize - padSize) * sizeof(float), padSize, padSize);
	colLo.srcPos = make_cudaPos(0 * sizeof(float), paddedFieldSize - 2 * padSize, padSize);
	colLo.dstPos = make_cudaPos(0 * sizeof(float), 0, padSize);
	colHi.srcPos = make_cudaPos(0 * sizeof(float), padSize, padSize);
	colHi.dstPos = make_cudaPos(0 * sizeof(float), paddedFieldSize - padSize, padSize);
	layLo.srcPos = make_cudaPos(0 * sizeof(float), 0, paddedFieldSize - 2 * padSize);
	layLo.dstPos = make_cudaPos(0 * sizeof(float), 0, 0);
	layHi.srcPos = make_cudaPos(0 * sizeof(float), 0, padSize);
	layHi.dstPos = make_cudaPos(0 * sizeof(float), 0, paddedFieldSize - padSize);

	rowLo.kind = cudaMemcpyDeviceToDevice;
	rowHi.kind = cudaMemcpyDeviceToDevice;
	colLo.kind = cudaMemcpyDeviceToDevice;
	colHi.kind = cudaMemcpyDeviceToDevice;
	layLo.kind = cudaMemcpyDeviceToDevice;
	layHi.kind = cudaMemcpyDeviceToDevice;
};
