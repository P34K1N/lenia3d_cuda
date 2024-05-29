import cupy as cp
    

GROWTH_TEMPLATE = """
extern "C"
__global__ void {0}(float* growth, int fieldSize, int order, int asyncRadius) {{
	int asyncRadiusSqr = asyncRadius * asyncRadius;
	int row = asyncRadius * (blockIdx.x * blockDim.x + threadIdx.x) + (order % asyncRadius);
	int col = asyncRadius * (blockIdx.y * blockDim.y + threadIdx.y) + (order % asyncRadiusSqr / asyncRadius);
	int lay = asyncRadius * (blockIdx.z * blockDim.z + threadIdx.z) + (order / asyncRadiusSqr);

	if (row < fieldSize && col < fieldSize && lay < fieldSize) {{
		int fieldSizeSqr = fieldSize * fieldSize;
		int idx = lay * fieldSizeSqr + col * fieldSize + row;

		float x = growth[idx];
		{1}
		growth[idx] = x;
	}}
}}
"""


class GrowthFuncConstructor(object):
    @staticmethod
    def PreCompile(func):
        x = cp.zeros((1, 1, 1), dtype=cp.float32)
        func((1,1,1), (1,1,1), (x, 1, 0, 1))

    @staticmethod
    def FromExpr(expression, name='custom_growth_func', precompile=True):
        code = GROWTH_TEMPLATE.format(name, expression)
        func = cp.RawKernel(code, name)
        if (precompile):
            GrowthFuncConstructor.PreCompile(func)
        return func

    @staticmethod
    def Exponential(mu, sigma):
        return GrowthFuncConstructor.FromExpr(
            "x = 2.0 * expf( - (x - {0}) * (x - {0}) / (2.0 * {1} * {1})) - 1.0;".format(mu, sigma),
            name = 'exponential_growth_func'
        )
    
    @staticmethod
    def Polynomial(mu, sigma, alpha=4):
        return GrowthFuncConstructor.FromExpr(
            """
            if (x >= {0} - 3.0 * {1} && x <= {0} + 3.0 * {1}) {{
                x = 2.0 * powf(1.0 - (x - {0}) * (x - {0}) / (9.0 * {1} * {1}), {2}) - 1.0;
            }}
            else {{
                x = -1.0;
            }}
            """.format(mu, sigma, alpha),
            name = 'polynomial_growth_func'
        )
    
    @staticmethod
    def Rectangular(mu, sigma):
        return GrowthFuncConstructor.FromExpr(
            """
            if (x >= {0} - {1} && x <= {0} + {1}) {{
                x = 1.0;
            }}
            else {{
                x = -1.0;
            }}
            """.format(mu, sigma),
            name = 'rectangular_growth_func'
        )
    
    @staticmethod
    def Identity():
        return GrowthFuncConstructor.FromExpr(
            "x = x;",
            name = 'identity_growth_func'
        )
    