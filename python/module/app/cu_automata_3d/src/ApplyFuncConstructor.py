import cupy as cp

    
APPLY_TEMPLATE = """
extern "C"
__global__ void {0}(float* img, float* growth,
	int fieldSize, int padSize, int paddedFieldSize, double timeDelta,
    float* randomValues, int order, int asyncRadius) {{
	int asyncRadiusSqr = asyncRadius * asyncRadius;
	int row = asyncRadius * (blockIdx.x * blockDim.x + threadIdx.x) + (order % asyncRadius);
	int col = asyncRadius * (blockIdx.y * blockDim.y + threadIdx.y) + (order % asyncRadiusSqr / asyncRadius);
	int lay = asyncRadius * (blockIdx.z * blockDim.z + threadIdx.z) + (order / asyncRadiusSqr);

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {{
		int paddedFieldSizeSqr = paddedFieldSize * paddedFieldSize;
		int fieldSizeSqr = fieldSize * fieldSize;

		int imgRow = row + padSize;
		int imgCol = col + padSize;
		int imgLay = lay + padSize;

		int imgIdx = imgLay * paddedFieldSizeSqr + imgCol * paddedFieldSize + imgRow;
        float a = img[imgIdx];
        float g = growth[lay * fieldSizeSqr + col * fieldSize + row];
        float dt = timeDelta;
        float xi = randomValues[lay * fieldSizeSqr + col * fieldSize + row];
        float res;

        {1}

		img[imgIdx] = res;
	}}
}}
"""


class ApplyFuncConstructor(object):
    @staticmethod
    def PreCompile(func):
        x = cp.zeros((3, 3, 3), dtype=cp.float32)
        y = cp.zeros((1, 1, 1), dtype=cp.float32)
        func((1,1,1), (1,1,1), (y, x, 1, 1, 3, 1, x, 0, 1))

    @staticmethod
    def FromExpr(expression, name='custom_apply_func', precompile=True):
        code = APPLY_TEMPLATE.format(name, expression)
        func = cp.RawKernel(code, name)
        if (precompile):
            ApplyFuncConstructor.PreCompile(func)
        return func
    
    @staticmethod
    def Increment():
        return ApplyFuncConstructor.FromExpr(
            """
            res = a + dt * g;
		    res = res > 1.0 ? 1.0 : (res < 0.0 ? 0.0 : res);
            """,
            name = 'increment_apply_func'
        )
    
    @staticmethod
    def Replace():
        return ApplyFuncConstructor.FromExpr(
            """
            res = g;
            """,
            name = 'replace_apply_func'
        )
    
    @staticmethod
    def Ising(beta):
        return ApplyFuncConstructor.FromExpr(
            """
            res = (a * g <= 0 || xi < expf(-2 * {0} * a * g)) ? -a : a;
            """.format(beta),
            name = 'ising_apply_func'
        )
    