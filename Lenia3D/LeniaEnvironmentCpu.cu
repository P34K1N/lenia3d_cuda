#include "LeniaEnvironmentCpu.cuh"
#include "KernelFunctions.cuh"

#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <fstream>

LeniaEnvironmentCpu::LeniaEnvironmentCpu(std::string file, int fieldSize, int kernelRadius, int stepsPerTimeUnit, float mu, float sigma,
	KernelType kernelType, GrowthType growthType) :
	fieldSize(fieldSize), kernelSize(2 * kernelRadius + 1), padSize(kernelRadius),
	paddedFieldSize(fieldSize + 2 * padSize),
	volume(fieldSize * fieldSize * fieldSize),
	volumePadded(paddedFieldSize * paddedFieldSize * paddedFieldSize),
	volumeKernel(kernelSize * kernelSize * kernelSize),
	mu(mu), sigma(sigma), 
	stepsPerTimeUnit(stepsPerTimeUnit), timeDelta(1.0 / stepsPerTimeUnit),
	growthType(growthType) {

	// malloc arrays
	imgCpu = (float*)calloc(volume, sizeof(*imgCpu));
	kernel = (float*)calloc(volumeKernel, sizeof(*kernel));
	growth = (float*)calloc(volume, sizeof(*growth));

	// initialize field
	std::ifstream input(file, std::ios::binary);
	input.read((char *)imgCpu, volume * sizeof(*imgCpu));

	// initialize convolution kernel

	// malloc array for cpu kernel
	float* cpuKernelTmp = (float*)calloc(volumeKernel, sizeof(*cpuKernelTmp));

	// calculate kernel
	switch (kernelType) {
	case KernelType::kUniform:
		SetUniformKernel();
		break;
	case KernelType::kMove:
		SetMoveKernel();
		break;
	case KernelType::kExponential:
		SetExponentialKernel();
		break;
	case KernelType::kPolynomial:
		SetPolynomialKernel();
		break;
	case KernelType::kRectangular:
		SetRectangularKernel();
		break;
	case KernelType::kGol:
		SetGolKernel();
		break;
	}
};

void LeniaEnvironmentCpu::NormalizeKernel() {
	float sum = 0.0;
	for (int idx = 0; idx < volumeKernel; idx++) {
		sum += kernel[idx];
		// std::cout << "sum += kernel[" << idx << "] = " << kernel[idx] << std::endl;
	}
	for (int idx = 0; idx < volumeKernel; idx++) {
		kernel[idx] /= sum;
		// std::cout << "kernel[" << idx << "] / " << sum << " = " << kernel[idx] << std::endl;
	}
}

void LeniaEnvironmentCpu::SetUniformKernel() {
	for (int idx = 0; idx < volumeKernel; idx++)
		kernel[idx] = 1.0;
	NormalizeKernel();
}

void LeniaEnvironmentCpu::SetMoveKernel() {
	kernel[0] = 1.0;
	NormalizeKernel();
}

void LeniaEnvironmentCpu::SetExponentialKernel() {
	int mid = padSize;
	int padSizeSqr = padSize * padSize;
	for (int lay = 0; lay < kernelSize; lay++) {
		float laySqr = (lay - mid) * (lay - mid);
		for (int col = 0; col < kernelSize; col++) {
			float colSqr = (col - mid) * (col - mid);
			for (int row = 0; row < kernelSize; row++) {
				float rowSqr = (row - mid) * (row - mid);
				float r = sqrt(laySqr + colSqr + rowSqr) / mid;
				if (r < 1) {
					kernel[lay * padSizeSqr + col * padSize + row] = 
						exp(4 - 1 / (r * (1 - r)));
				}
			}
		}
	}
	NormalizeKernel();
}

void LeniaEnvironmentCpu::SetPolynomialKernel() {
	int mid = padSize;
	int padSizeSqr = padSize * padSize;
	for (int lay = 0; lay < kernelSize; lay++) {
		float laySqr = (lay - mid) * (lay - mid);
		for (int col = 0; col < kernelSize; col++) {
			float colSqr = (col - mid) * (col - mid);
			for (int row = 0; row < kernelSize; row++) {
				float rowSqr = (row - mid) * (row - mid);
				float r = sqrt(laySqr + colSqr + rowSqr);
				if (r < 1) {
					kernel[lay * padSizeSqr + col * padSize + row] = 
						 pow(4 * r * (1 - r), 4);
				}
			}
		}
	}
	NormalizeKernel();
}

void LeniaEnvironmentCpu::SetRectangularKernel() {
	int mid = padSize;
	int padSizeSqr = padSize * padSize;
	for (int lay = 0; lay < kernelSize; lay++) {
		float laySqr = (lay - mid) * (lay - mid);
		for (int col = 0; col < kernelSize; col++) {
			float colSqr = (col - mid) * (col - mid);
			for (int row = 0; row < kernelSize; row++) {
				float rowSqr = (row - mid) * (row - mid);
				float r = sqrt(laySqr + colSqr + rowSqr);
				if (r < 1) {
					kernel[lay * padSizeSqr + col * padSize + row] = 
						 (r >= 0.25 && r <= 0.75) ? 1.0 : 0.0;
				}
			}
		}
	}
	NormalizeKernel();
}

void LeniaEnvironmentCpu::SetGolKernel() {
	int mid = padSize;
	int padSizeSqr = padSize * padSize;
	for (int lay = 0; lay < kernelSize; lay++) {
		float laySqr = (lay - mid) * (lay - mid);
		for (int col = 0; col < kernelSize; col++) {
			float colSqr = (col - mid) * (col - mid);
			for (int row = 0; row < kernelSize; row++) {
				float rowSqr = (row - mid) * (row - mid);
				float r = sqrt(laySqr + colSqr + rowSqr);
				if (r < 1) {
					kernel[lay * padSizeSqr + col * padSize + row] = 
						 (r >= 0.25 && r <= 0.87) ? 1.0 : (r < 0.25 ? 0.5 : 0.0);
				}
			}
		}
	}
	NormalizeKernel();
}

void LeniaEnvironmentCpu::Convolute() {
	for (int lay = 0; lay < fieldSize; lay++) {
		for (int col = 0; col < fieldSize; col++) {
			for (int row = 0; row < fieldSize; row++) {
				float sum = 0.0;

				int fieldSizeSqr = fieldSize * fieldSize;
				int kernelSizeSqr = kernelSize * kernelSize;

				for (int kernelLay = 0; kernelLay < kernelSize; kernelLay++) {
					int imgLay = lay + kernelLay - padSize;
					if (imgLay <= 0) {
						imgLay += fieldSize;
					}
					if (imgLay >= fieldSize) {
						imgLay -= fieldSize;
					}
					for (int kernelCol = 0; kernelCol < kernelSize; kernelCol++) {
						int imgCol = col + kernelCol - padSize;
						if (imgCol <= 0) {
							imgCol += fieldSize;
						}
						if (imgCol >= fieldSize) {
							imgCol -= fieldSize;
						}
						for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
							int imgRow = row + kernelRow - padSize;
							if (imgRow <= 0) {
								imgRow += fieldSize;
							}
							if (imgRow >= fieldSize) {
								imgRow -= fieldSize;
							}

							// printf("res[%d][%d][%d]: img[%d][%d][%d](%f) * ker[%d][%d][%d](%f)\n", lay, col, row,
							// 	imgLay, imgCol, imgRow,
							// 	imgCpu[imgLay * fieldSizeSqr + imgCol * fieldSize + imgRow],
							// 	kernelLay, kernelCol, kernelRow,
							// 	kernel[kernelLay * kernelSizeSqr + kernelCol * kernelSize + kernelRow]);
							sum +=
								imgCpu[imgLay * fieldSizeSqr + imgCol * fieldSize + imgRow] *
								kernel[kernelLay * kernelSizeSqr + kernelCol * kernelSize + kernelRow];
						}
					}
				}
				// printf("growth[%d][%d][%d] = %f\n",
				// 	lay, col, row, sum);
				growth[lay * fieldSizeSqr +
					col * fieldSize +
					row] = sum;
			}
		}
	}
}

void LeniaEnvironmentCpu::ApplyDoubleGrowth() {
	int fieldSizeSqr = fieldSize * fieldSize;
	for (int lay = 0; lay < fieldSize; lay++) {
		for (int col = 0; col < fieldSize; col++) {
			for (int row = 0; row < fieldSize; row++) {
				int idx = lay * fieldSizeSqr + col * fieldSize + row;
				growth[idx] *= 2;
			}
		}
	}
}

void LeniaEnvironmentCpu::ApplyExponentialGrowth() {
	int fieldSizeSqr = fieldSize * fieldSize;
	for (int lay = 0; lay < fieldSize; lay++) {
		for (int col = 0; col < fieldSize; col++) {
			for (int row = 0; row < fieldSize; row++) {
				int idx = lay * fieldSizeSqr + col * fieldSize + row;
				float orig = growth[idx];
				orig = 2.0 * expf( - (orig - mu) * (orig - mu) / (2.0 * sigma * sigma)) - 1.0;
				growth[idx] = orig;
			}
		}
	}
}

void LeniaEnvironmentCpu::ApplyPolynomialGrowth() {
	int fieldSizeSqr = fieldSize * fieldSize;
	for (int lay = 0; lay < fieldSize; lay++) {
		for (int col = 0; col < fieldSize; col++) {
			for (int row = 0; row < fieldSize; row++) {
				int idx = lay * fieldSizeSqr + col * fieldSize + row;
				float orig = growth[idx];
				if (orig >= mu - 3.0 * sigma && orig <= mu + 3.0 * sigma) {
					orig = 2.0 * powf(1.0 - (orig - mu) * (orig - mu) / (9.0 * sigma * sigma), 4.0) - 1.0;
				}
				else {
					orig = -1.0;
				}
				growth[idx] = orig;
			}
		}
	}
}

void LeniaEnvironmentCpu::ApplyRectangularGrowth() {
	int fieldSizeSqr = fieldSize * fieldSize;
	for (int lay = 0; lay < fieldSize; lay++) {
		for (int col = 0; col < fieldSize; col++) {
			for (int row = 0; row < fieldSize; row++) {
				int idx = lay * fieldSizeSqr + col * fieldSize + row;
				float orig = growth[idx];
				if (orig >= mu - sigma && orig <= mu + sigma) {
					orig = 1.0;
				}
				else {
					orig = -1.0;
				}
				growth[idx] = orig;
			}
		}
	}
}

void LeniaEnvironmentCpu::ApplyGrowth() {
	switch (growthType) {
	case GrowthType::kDouble:
		ApplyDoubleGrowth();
		break;
	case GrowthType::kExponential:
		ApplyExponentialGrowth();
		break;
	case GrowthType::kPolynomial:
		ApplyPolynomialGrowth();
		break;
	case GrowthType::kRectangular:
		ApplyRectangularGrowth();
		break;
	}
}

void LeniaEnvironmentCpu::SumAndCut() {
	int fieldSizeSqr = fieldSize * fieldSize;
	for (int lay = 0; lay < fieldSize; lay++) {
		for (int col = 0; col < fieldSize; col++) {
			for (int row = 0; row < fieldSize; row++) {
				int idx = lay * fieldSizeSqr + col * fieldSize + row;
				float sum = imgCpu[idx]
					+ timeDelta * growth[idx];
				sum = sum > 1.0 ? 1.0 : (sum < 0.0 ? 0.0 : sum);
				imgCpu[idx] = sum;
			}
		}
	}
}

void LeniaEnvironmentCpu::Step() {
	Convolute();
	ApplyGrowth();
	SumAndCut();
}

float* LeniaEnvironmentCpu::GetResult() {
	return imgCpu;
}

float* LeniaEnvironmentCpu::StepTimeUnit() {
	for (int i = 0; i < stepsPerTimeUnit; i++) {
		Step();
	}
	return GetResult();
}

void LeniaEnvironmentCpu::PutStepTimeUnit(std::string file) {
	float* result = StepTimeUnit();
	std::ofstream output(file, std::ios::binary);
	output.write((char *)result, volume * sizeof(*result));
}

void LeniaEnvironmentCpu::PutResult(std::string file) {
	float* result = GetResult();
	std::ofstream output(file, std::ios::binary);
	output.write((char*)result, volume * sizeof(*result));
}


LeniaEnvironmentCpu::~LeniaEnvironmentCpu() {
	free(imgCpu);
	free(growth);
	free(kernel);
};
