import math

import cupy as cp

class KernelConstructor(object):
    @staticmethod
    def Normalize(kernel):
        sum = kernel.sum(-1).sum(-1).sum(-1)
        kernel /= sum

    @staticmethod
    def FromFunc(radius, func, normalize=True):
        size = radius * 2 + 1
        kernel = cp.zeros((size, size, size), dtype=cp.float32)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    kernel[i][j][k] = func(i, j, k)
        if normalize:
            KernelConstructor.Normalize(kernel)
        return kernel
    
    @staticmethod
    def FromRadialFunc(radius, func, normalize=True):
        mid = radius

        def _func(i, j, k):
            r = math.sqrt(
                (i - mid) * (i - mid) +
                (j - mid) * (j - mid) +
                (k - mid) * (k - mid)
            ) / mid
            if (r > 1 or r == 0):
                return 0.0
            return func(r)
        
        return KernelConstructor.FromFunc(radius, _func, normalize)
    
    @staticmethod
    def Shell(func, beta):
        def _shell(r):
            br = len(beta) * r
            mod, flr = math.modf(br)
            if (mod == 0 or mod == 1):
                return 0.0
            return beta[int(flr)] * func(mod)

        return _shell


    @staticmethod
    def Exponential(radius, alpha=4, beta=[1]):
        def _expfunc(r):
            return math.exp(alpha - alpha / (4 * r * (1 - r)))
        
        func = KernelConstructor.Shell(_expfunc, beta)
        return KernelConstructor.FromRadialFunc(radius, func)

    @staticmethod
    def Polynomial(radius, alpha=4, beta=[1]):
        def _polyfunc(r):
            return (4 * r * (1 - r)) ** alpha

        func = KernelConstructor.Shell(_polyfunc, beta)
        return KernelConstructor.FromRadialFunc(radius, func)
    
    @staticmethod
    def Rectangular(radius, cut=(0.25, 0.75), beta=[1]):
        def _rectfunc(r):
            return 1.0 if (r >= cut[0] and r <= cut[1]) else 0.0

        func = KernelConstructor.Shell(_rectfunc, beta)
        return KernelConstructor.FromRadialFunc(radius, func)
    
    @staticmethod
    def Identity(radius):
        mid = radius

        def _identityfunc(i, j, k):
            return 1.0 if (i == mid and j == mid and k == mid) else 0.0

        return KernelConstructor.FromFunc(radius, _identityfunc)
    
    @staticmethod
    def DiagShift(radius):
        mid = radius

        def _shiftfunc(i, j, k):
            if (i == mid and j == mid and k == mid):
                return -1.0
            if (i == mid + 1 and j == mid + 1 and k == mid + 1):
                return 1.0
            return 0.0

        return KernelConstructor.FromFunc(radius, _shiftfunc, normalize=False)
    
    @staticmethod
    def Ising(infl_radius, J, scale=3):
        radius = math.ceil(infl_radius)

        def __isingfunc(r):
            R = r * radius
            if R > infl_radius:
                return 0
            if R == 0:
                return 0
            return J / (R**scale)
    
        return KernelConstructor.FromRadialFunc(radius, __isingfunc, normalize=False)