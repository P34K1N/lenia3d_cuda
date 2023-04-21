#include "KernelFunctions.cuh"

#include <iostream>

namespace {
	__device__ void normalize(float* kernel, int kernelVolume, float* sumReductionBuf, int sumReductionBufSize, int idx) {
		__syncthreads();
		
		int step = sumReductionBufSize;

		if (idx < step && idx >= 0) {
			sumReductionBuf[idx] = kernel[idx];
			// printf("sumReductionBuf[%d] = %f\n", idx, sumReductionBuf[idx]);
			if (idx + step < kernelVolume) {
				sumReductionBuf[idx] += kernel[idx + step];
				// printf("sumReductionBuf[%d] += %f\n", idx, kernel[idx + step]);
			}
			// printf("sumReductionBuf[%d] = %f\n", idx, sumReductionBuf[idx]);
			
		}
		__syncthreads();

		step >>= 1;

		while (step > 0) {
			if (idx < step && idx >= 0) {
				sumReductionBuf[idx] += sumReductionBuf[idx + step];
			}

			step >>= 1;
			__syncthreads();
		}

		float sum = sumReductionBuf[0];
		if (idx < kernelVolume && idx >= 0) {
			kernel[idx] /= sum;
		//	printf("kernel[%d] / %f = %f\n",
		//		idx, sum, kernel[idx]);
		}
	}
}

__global__ void convolute(float* img, float* kernel, float* growth,
	int paddedFieldSize, int kernelSize, int fieldSize)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {
		float sum = 0.0;

		int paddedFieldSizeSqr = paddedFieldSize * paddedFieldSize;
		int fieldSizeSqr = fieldSize * fieldSize;
		int kernelSizeSqr = kernelSize * kernelSize;

		for (int kernelLay = 0; kernelLay < kernelSize; kernelLay++) {
			for (int kernelCol = 0; kernelCol < kernelSize; kernelCol++) {
				for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
					int imgRow = row + kernelRow;
					int imgCol = col + kernelCol;
					int imgLay = lay + kernelLay;

					//printf("res[%d][%d][%d]: img[%d][%d][%d](%f) * ker[%d][%d][%d](%f)\n", lay, col, row,
					//	imgPad, imgCol, imgRow,
					//	image[imgPad * imageRowsCols + imgCol * imageRows + imgRow],
					//	kernelLay, kernelCol, kernelRow,
					//	kernel[kernelLay * kernelRowsCols + kernelCol * kernelRows + kernelRow]);
					sum +=
						img[imgLay * paddedFieldSizeSqr + imgCol * paddedFieldSize + imgRow] *
						kernel[kernelLay * kernelSizeSqr + kernelCol * kernelSize + kernelRow];
				}
			}
		}
		//printf("growth[%d][%d][%d] = %f\n",
		//	lay, col, row, sum);
		growth[lay * fieldSizeSqr +
			col * fieldSize +
			row] = sum;
	}
}

__global__ void sumAndCut(float* img, float* growth,
	int fieldSize, int padSize, int paddedFieldSize, float timeDelta) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {
		int paddedFieldSizeSqr = paddedFieldSize * paddedFieldSize;
		int fieldSizeSqr = fieldSize * fieldSize;

		int imgRow = row + padSize;
		int imgCol = col + padSize;
		int imgLay = lay + padSize;

		int imgIdx = imgLay * paddedFieldSizeSqr + imgCol * paddedFieldSize + imgRow;

		float sum = img[imgIdx]
			+ timeDelta * growth[lay * fieldSizeSqr + col * fieldSize + row];
		sum = sum > 1.0 ? 1.0 : (sum < 0.0 ? 0.0 : sum);
		img[imgIdx] = sum;
		//printf("img[%d][%d][%d] (aka img[%d]) += growth[%d][%d][%d](%f)\n", 
		//	imgLay, imgCol, imgRow, imgIdx, lay, col, row, growth[lay * fieldSizeSqr + col * fieldSize + row]);
		//img[imgIdx] = growth[lay * fieldSizeSqr + col * fieldSize + row];
	}
}

__global__ void setUniformKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = -1;

	if (row < kernelDim && col < kernelDim && lay < kernelDim) {
		idx = lay * kernelDim * kernelDim + col * kernelDim + row;
		kernel[idx] 
			= 1.;
	}

	normalize(kernel, kernelDim * kernelDim * kernelDim, sumReductionBuf, sumReductionBufSize, idx);
}

__global__ void setMoveKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = -1;

	if (row == 0 && col == 0 && lay == 0) {
		idx = lay * kernelDim * kernelDim + col * kernelDim + row;
		kernel[idx]
			= 1.;
	}

	normalize(kernel, kernelDim * kernelDim * kernelDim, sumReductionBuf, sumReductionBufSize, idx);
}

__global__ void setExponentialKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = -1;

	if (row < kernelDim && col < kernelDim && lay < kernelDim) {
		idx = lay * kernelDim * kernelDim + col * kernelDim + row;
		float radius = (kernelDim - 1) / 2;
		float r = sqrtf((row - radius) * (row - radius)
			+ (col - radius) * (col - radius)
			+ (lay - radius) * (lay - radius)) / radius;
		if (r > 1) {
			kernel[idx]
				= expf(4 - 1 / (r * (1 - r)));
		}
	}

	normalize(kernel, kernelDim * kernelDim * kernelDim, sumReductionBuf, sumReductionBufSize, idx);
}

__global__ void setPolynomialKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = -1;

	if (row < kernelDim && col < kernelDim && lay < kernelDim) {
		idx = lay * kernelDim * kernelDim + col * kernelDim + row;
		float radius = (kernelDim - 1) / 2;
		float r = sqrtf((row - radius) * (row - radius)
			+ (col - radius) * (col - radius)
			+ (lay - radius) * (lay - radius)) / radius;
		if (r > 1) {
			kernel[idx]
				= powf(4 * r * (1 - r), 4);
		}
	}

	normalize(kernel, kernelDim * kernelDim * kernelDim, sumReductionBuf, sumReductionBufSize, idx);
}

__global__ void setRectangularKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = -1;

	if (row < kernelDim && col < kernelDim && lay < kernelDim) {
		idx = lay * kernelDim * kernelDim + col * kernelDim + row;
		float radius = (kernelDim - 1) / 2;
		float r = sqrtf((row - radius) * (row - radius)
			+ (col - radius) * (col - radius)
			+ (lay - radius) * (lay - radius)) / radius;
		if (r > 1) {
			kernel[idx]
				= (r >= 0.25 && r <= 0.75) ? 1.0 : 0.0;
		}
	}

	normalize(kernel, kernelDim * kernelDim * kernelDim, sumReductionBuf, sumReductionBufSize, idx);
}

__global__ void setGolKernel(float* kernel, int kernelDim, float* sumReductionBuf, int sumReductionBufSize) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;
	int idx = -1;

	if (row < kernelDim && col < kernelDim && lay < kernelDim) {
		idx = lay * kernelDim * kernelDim + col * kernelDim + row;
		float radius = (kernelDim - 1) / 2;
		float r = sqrtf((row - radius) * (row - radius) 
			+ (col - radius) * (col - radius) 
			+ (lay - radius) * (lay - radius)) / radius;
		if (r <= 1) {
			kernel[idx]
				= (r >= 0.25 && r <= 0.87) ? 1.0 : (r < 0.25 ? 0.5 : 0.0);
			//printf("kernel[%d][%d][%d] (rad = %f) = %f\n", 
			//	lay, col, row, r, kernel[idx]);
		}
	}

	normalize(kernel, kernelDim * kernelDim * kernelDim, sumReductionBuf, sumReductionBufSize, idx);
}

__global__ void applyDoubleGrowth(float* growth, int fieldSize, float mu, float sigma) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {
		int fieldSizeSqr = fieldSize * fieldSize;
		int idx = lay * fieldSizeSqr + col * fieldSize + row;

		float orig = growth[idx];
		orig *= 2;
		growth[idx] = orig;
	}
}

__global__ void applyExponentialGrowth(float* growth, int fieldSize, float mu, float sigma) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {
		int fieldSizeSqr = fieldSize * fieldSize;
		int idx = lay * fieldSizeSqr + col * fieldSize + row;

		float orig = growth[idx];
		orig = 2.0 * expf( - (orig - mu) * (orig - mu) / (2.0 * sigma * sigma)) - 1.0;
		growth[idx] = orig;
	}
}

__global__ void applyPolynomialGrowth(float* growth, int fieldSize, float mu, float sigma) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {
		int fieldSizeSqr = fieldSize * fieldSize;
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

__global__ void applyRectangularGrowth(float* growth, int fieldSize, float mu, float sigma) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int lay = blockIdx.z * blockDim.z + threadIdx.z;

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {
		int fieldSizeSqr = fieldSize * fieldSize;
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
