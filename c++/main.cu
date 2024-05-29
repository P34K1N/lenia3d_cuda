#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>						// cudaDeviceSynchronize()
#include <iostream>
#include <iomanip>
#include <chrono>

#include "LeniaEnvironment.cuh"
#include "EnvironmentCpu.cuh"

void printResult(float* result, int fieldSize) {
	for (size_t lay = 0; lay < fieldSize; lay++) {
		for (size_t col = 0; col < fieldSize; col++)
		{
			for (size_t row = 0; row < fieldSize; row++)
			{
				std::cout << std::fixed << std::setw(3) << std::setprecision(1)
					<< result[lay * fieldSize * fieldSize + col * fieldSize + row] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char ** argv) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	int fieldSize = std::stoi(argv[1]);
	int kernelRadius = std::stoi(argv[2]);
	int stepsPerTimeUnit = std::stoi(argv[3]);
	int timeUnits = std::stoi(argv[4]);
	float mu = std::stof(argv[5]);
	float sigma = std::stof(argv[6]);

	if (argv[10][0] == 'g') {
		LeniaEnvironment::KernelType kernelType;
		switch (argv[7][0]) {
		case 'u':
			kernelType = LeniaEnvironment::KernelType::kUniform;
			break;
		case 'm':
			kernelType = LeniaEnvironment::KernelType::kMove;
			break;
		case 'e':
			kernelType = LeniaEnvironment::KernelType::kExponential;
			break;
		case 'p':
			kernelType = LeniaEnvironment::KernelType::kPolynomial;
			break;
		case 'r':
			kernelType = LeniaEnvironment::KernelType::kRectangular;
			break;
		case 'g':
			kernelType = LeniaEnvironment::KernelType::kGol;
			break;
		default:
			std::cerr << "Unknown kernel type: " << argv[7][0] << std::endl;
			throw std::runtime_error("Unknown kernel type");
		}

		LeniaEnvironment::GrowthType growthType;
		switch (argv[8][0]) {
		case 'd':
			growthType = LeniaEnvironment::GrowthType::kDouble;
			break;
		case 'e':
			growthType = LeniaEnvironment::GrowthType::kExponential;
			break;
		case 'p':
			growthType = LeniaEnvironment::GrowthType::kPolynomial;
			break;
		case 'r':
			growthType = LeniaEnvironment::GrowthType::kRectangular;
			break;
		default:
			std::cerr << "Unknown growth type" << std::endl;
			throw std::runtime_error("Unknown growth type");
		}

		LeniaEnvironment env(argv[9], fieldSize, kernelRadius, stepsPerTimeUnit, mu, sigma,
			kernelType, growthType);

		int num_width = int(ceilf(log10f(timeUnits + 1)));
		std::string file = "res/" + std::string(num_width, '0') + ".bin";
		env.PutResult(file);

		for (int i = 0; i < timeUnits; i++) {
			std::string idx = std::to_string(i + 1);
			file = "res/" + std::string(num_width - idx.length(), '0') + idx + ".bin";
			env.PutStepTimeUnit(file);
		}
	}
	if (argv[10][0] == 'c') {
		EnvironmentCpu::KernelType kernelType;
		switch (argv[7][0]) {
		case 'u':
			kernelType = EnvironmentCpu::KernelType::kUniform;
			break;
		case 'm':
			kernelType = EnvironmentCpu::KernelType::kMove;
			break;
		case 'e':
			kernelType = EnvironmentCpu::KernelType::kExponential;
			break;
		case 'p':
			kernelType = EnvironmentCpu::KernelType::kPolynomial;
			break;
		case 'r':
			kernelType = EnvironmentCpu::KernelType::kRectangular;
			break;
		case 'g':
			kernelType = EnvironmentCpu::KernelType::kGol;
			break;
		default:
			std::cerr << "Unknown kernel type: " << argv[7][0] << std::endl;
			throw std::runtime_error("Unknown kernel type");
		}

		EnvironmentCpu::GrowthType growthType;
		switch (argv[8][0]) {
		case 'd':
			growthType = EnvironmentCpu::GrowthType::kDouble;
			break;
		case 'e':
			growthType = EnvironmentCpu::GrowthType::kExponential;
			break;
		case 'p':
			growthType = EnvironmentCpu::GrowthType::kPolynomial;
			break;
		case 'r':
			growthType = EnvironmentCpu::GrowthType::kRectangular;
			break;
		default:
			std::cerr << "Unknown growth type" << std::endl;
			throw std::runtime_error("Unknown growth type");
		}

		EnvironmentCpu env(argv[9], fieldSize, kernelRadius, stepsPerTimeUnit, mu, sigma,
			kernelType, growthType);

		int num_width = int(ceilf(log10f(timeUnits + 1)));
		std::string file = "res/" + std::string(num_width, '0') + ".bin";
		env.PutResult(file);

		for (int i = 0; i < timeUnits; i++) {
			std::string idx = std::to_string(i + 1);
			file = "res/" + std::string(num_width - idx.length(), '0') + idx + ".bin";
			env.PutStepTimeUnit(file);
		}
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Elapsed time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0 << " s" << std::endl;
}