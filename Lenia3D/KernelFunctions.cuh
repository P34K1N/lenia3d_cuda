#pragma once
#include "device_launch_parameters.h"

__global__ void convolute(float* img, float* kernel, float* growth,
	int paddedFieldSize, int kernelSize, int fieldSize);
__global__ void sumAndCut(float* img, float* growth,
	int fieldSize, int padSize, int paddedFieldSize, float timeDelta);

__global__ void setUniformKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize);
__global__ void setMoveKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize);
__global__ void setExponentialKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize);
__global__ void setPolynomialKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize);
__global__ void setRectangularKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize);
__global__ void setGolKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize);

__global__ void applyDoubleGrowth(float* growth, int fieldSize, float mu, float sigma);
__global__ void applyExponentialGrowth(float* growth, int fieldSize, float mu, float sigma);
__global__ void applyPolynomialGrowth(float* growth, int fieldSize, float mu, float sigma);
__global__ void applyRectangularGrowth(float* growth, int fieldSize, float mu, float sigma);