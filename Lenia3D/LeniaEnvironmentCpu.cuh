#pragma once

#include <math.h>
#include <cuda_runtime_api.h>	
#include <string>
#include "device_launch_parameters.h"

class LeniaEnvironmentCpu {
public:
	enum class KernelType {
		kUniform, kMove, kExponential, kPolynomial, kRectangular, kGol
	};
	enum class GrowthType {
		kDouble, kExponential, kPolynomial, kRectangular
	};

	LeniaEnvironmentCpu(std::string file, int fieldSize, int kernelRadius, int stepsPerTimeUnit, float mu, float sigma,
		KernelType kernelType, GrowthType growthType);
	~LeniaEnvironmentCpu();
	
	float* StepTimeUnit();
	float* GetResult();
	void PutStepTimeUnit(std::string file);
	void PutResult(std::string file);

private:
	void Convolute();
	void ApplyGrowth();
	void SumAndCut();
	void Step();

	void NormalizeKernel();
	void SetUniformKernel();
	void SetMoveKernel();
	void SetExponentialKernel();
	void SetPolynomialKernel();
	void SetRectangularKernel();
	void SetGolKernel();

	void ApplyDoubleGrowth();
	void ApplyExponentialGrowth();
	void ApplyPolynomialGrowth();
	void ApplyRectangularGrowth();

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
	float* growth;
	float* kernel;

	GrowthType growthType;
};