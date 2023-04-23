#pragma once

#include <math.h>
#include <cuda_runtime_api.h>	
#include <string>
#include "device_launch_parameters.h"

class LeniaEnvironment {
public:
	enum class KernelType {
		kUniform, kMove, kExponential, kPolynomial, kRectangular, kGol
	};
	enum class GrowthType {
		kDouble, kExponential, kPolynomial, kRectangular
	};

	LeniaEnvironment(std::string file, int fieldSize, int kernelRadius, int stepsPerTimeUnit, float mu, float sigma,
		KernelType kernelType, GrowthType growthType);
	~LeniaEnvironment();
	
	float* StepTimeUnit();
	float* GetResult();
	void PutStepTimeUnit(std::string file);
	void PutResult(std::string file);

private:
	void Convolute();
	void SetPadding();
	void ApplyGrowth();
	void SumAndCut();
	void Step();

	void NormalizeKernel(float* cpuKernelTmp);
	void SetUniformKernel(float* cpuKernelTmp);
	void SetMoveKernel(float* cpuKernelTmp);
	void SetExponentialKernel(float* cpuKernelTmp);
	void SetPolynomialKernel(float* cpuKernelTmp);
	void SetRectangularKernel(float* cpuKernelTmp);
	void SetGolKernel(float* cpuKernelTmp);

	int fieldSize;
	int kernelSize;
	int padSize;
	int paddedFieldSize;
	int volume;
	int volumePadded;
	int volumeKernel;
	
	int stepsPerTimeUnit;
	float timeDelta;

	float mu;
	float sigma;

	float* imgCpu;
	float* imgGpu;
	float* growth;
	float* kernel;
	
	dim3 gridDim;
	dim3 blockDim;

	struct MemcpyParams {
		MemcpyParams() = default;
		MemcpyParams(int fieldSize, int paddedFieldSize, int padSize, float* ptr);

		cudaMemcpy3DParms rowLo;
		cudaMemcpy3DParms rowHi;
		cudaMemcpy3DParms colLo;
		cudaMemcpy3DParms colHi;
		cudaMemcpy3DParms layLo;
		cudaMemcpy3DParms layHi;
	};
	MemcpyParams params;

	void (*applyGrowth)(float*, int, float, float);

	cudaError_t res;
};