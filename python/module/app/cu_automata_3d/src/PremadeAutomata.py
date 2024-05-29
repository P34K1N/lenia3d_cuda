import cupy as cp

from . import Environment, InitStateConstructor, KernelConstructor, GrowthFuncConstructor, ApplyFuncConstructor

class PremadeAutomata(object):
    @staticmethod
    def Ising(size, infl_radius, J, beta, up_pct=0.5, scale=3, async_mode=Environment.ASYNC_ORDERED_MODE):
        state = InitStateConstructor.InitStateConstructor.Choice(size, [1, -1], up_pct)
        kernel = KernelConstructor.KernelConstructor.Ising(infl_radius, J, scale)
        growth = GrowthFuncConstructor.GrowthFuncConstructor.Identity()
        apply = ApplyFuncConstructor.ApplyFuncConstructor.Ising(beta)
        env = Environment.Environment(state, kernel, growth, apply, asyncMode=async_mode)
        return env
    
    @staticmethod
    def Lenia(animal, kernel_type='e', growth_type='e', pad=0):
        state = InitStateConstructor.InitStateConstructor.FromLeniaRLE(animal['cells'])
        size = max(state.shape)
        state = cp.pad(
            state, 
            ((pad, pad + size - state.shape[0]), (pad, pad + size - state.shape[1]), (pad, pad + size - state.shape[2])), 
            'constant', 
            constant_values=(0, 0, 0, 0, 0, 0)
        )
        radius = animal['params']['R']
        stepsPerTimeUnit = animal['params']['T']

        def _fracture(strnum):
            if ('/' in strnum):
                vals = strnum.split('/')
                return int(vals[0]) / int(vals[1])
            return float(strnum)

        beta = [_fracture(n) for n in animal['params']['b'].split(',')]
        mu = animal['params']['m']
        sigma = animal['params']['s']

        if (kernel_type == 'e'):
            kernel = KernelConstructor.KernelConstructor.Exponential(radius, beta=beta)
        elif (kernel_type == 'p'):
            kernel = KernelConstructor.KernelConstructor.Polynomial(radius, beta=beta)
        elif (kernel_type == 'r'):
            kernel = KernelConstructor.KernelConstructor.Rectangular(radius, beta=beta)
        else:
            raise ValueError("kernel_type should be one of 'e', 'p', 'r', but got: {}".format(kernel_type))
        
        if (growth_type == 'e'):
            growth = GrowthFuncConstructor.GrowthFuncConstructor.Exponential(mu, sigma)
        elif (growth_type == 'p'):
            growth = GrowthFuncConstructor.GrowthFuncConstructor.Polynomial(mu, sigma)
        elif (growth_type == 'r'):
            growth = GrowthFuncConstructor.GrowthFuncConstructor.Rectangular(mu, sigma)
        else:
            raise ValueError("growth_type should be one of 'e', 'p', 'r', but got: {}".format(growth_type))
        
        apply = ApplyFuncConstructor.ApplyFuncConstructor.Increment()
        env = Environment.Environment(state, kernel, growth, apply, stepsPerTimeUnit)
        return env