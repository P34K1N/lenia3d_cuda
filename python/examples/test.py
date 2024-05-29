import cupy as cp

from Lenia3D.PythonModule.CuAutomata3D import Environment


if __name__ == '__main__':
    state = cp.random.rand(20, 20, 20, dtype=cp.float32)
    kernel = Environment.KernelConstructor.GetExponentialKernel(5)
    func = Environment.GrowthFuncConstructor.GetRectangularlGrowthFunc(0.33, 0.1)
    environ = Environment.LeniaEnvironment(state, kernel, func, 5)
    environ.StepTimeUnit()

