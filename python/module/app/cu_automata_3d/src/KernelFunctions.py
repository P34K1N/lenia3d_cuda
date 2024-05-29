import cupy as cp

CONVOUTE_CODE = """
extern "C"
__global__ void convolute(float* img, float* kernel, float* growth,
	int paddedFieldSize, int kernelSize, int fieldSize, int order, int asyncRadius)
{
	int asyncRadiusSqr = asyncRadius * asyncRadius;
	int row = asyncRadius * (blockIdx.x * blockDim.x + threadIdx.x) + (order % asyncRadius);
	int col = asyncRadius * (blockIdx.y * blockDim.y + threadIdx.y) + (order % asyncRadiusSqr / asyncRadius);
	int lay = asyncRadius * (blockIdx.z * blockDim.z + threadIdx.z) + (order / asyncRadiusSqr);

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

					sum +=
						img[imgLay * paddedFieldSizeSqr + imgCol * paddedFieldSize + imgRow] *
						kernel[kernelLay * kernelSizeSqr + kernelCol * kernelSize + kernelRow];
				}
			}
		}
		growth[lay * fieldSizeSqr +
			col * fieldSize +
			row] = sum;
	}
}
"""

convolute = cp.RawKernel(code=CONVOUTE_CODE, name='convolute')
convolute(
	(1,1,1), 
	(1,1,1), 
	(
		cp.zeros((3, 3, 3), dtype=cp.float32), 
		cp.zeros((1, 1, 1), dtype=cp.float32), 
		cp.zeros((1, 1, 1), dtype=cp.float32), 
		3, 
		1, 
		1,
		0, 
		1,
	)
)