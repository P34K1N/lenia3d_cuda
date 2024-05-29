import math
import random

import cupy as cp
from tqdm import tqdm

from . import KernelFunctions


THREADS_PER_BLOCK = 8
SYNC_MODE = 'sync'
ASYNC_ORDERED_MODE = 'async_ordered'
ASYNC_RANDOM_MODE = 'async_random'


class Environment(object):
    def SetPadding(self):
        self.img[0:self.padSize,:,:] = self.img[self.paddedFieldSize - 2 * self.padSize:self.paddedFieldSize - self.padSize,:,:]
        self.img[self.paddedFieldSize - self.padSize:self.paddedFieldSize,:,:] = self.img[self.padSize:2 * self.padSize,:,:]
        self.img[:,0:self.padSize,:] = self.img[:,self.paddedFieldSize - 2 * self.padSize:self.paddedFieldSize - self.padSize,:]
        self.img[:,self.paddedFieldSize - self.padSize:self.paddedFieldSize,:] = self.img[:,self.padSize:2 * self.padSize,:]
        self.img[:,:,0:self.padSize] = self.img[:,:,self.paddedFieldSize - 2 * self.padSize:self.paddedFieldSize - self.padSize]
        self.img[:,:,self.paddedFieldSize - self.padSize:self.paddedFieldSize] = self.img[:,:,self.padSize:2 * self.padSize]

    def Convolute(self, order, asyncRadius):
        KernelFunctions.convolute(
            self.gridDim, 
            self.blockDim, 
            (
                self.img, 
                self.kernel, 
                self.growth, 
                self.paddedFieldSize, 
                self.kernelSize, 
                self.fieldSize,
                order, 
                asyncRadius,
            )
        )

    def RunGrowth(self, order, asyncRadius):
        self.growthFunction(
            self.gridDim, 
            self.blockDim, 
            (
                self.growth, 
                self.fieldSize,
                order, 
                asyncRadius,
            )
        )

    def RunApply(self, order, asyncRadius):
        rand_vals = cp.random.rand(self.fieldSize, self.fieldSize, self.fieldSize).astype(cp.float32)
        self.applyFunction(
            self.gridDim, 
            self.blockDim, 
            (
                self.img,
                self.growth, 
                self.fieldSize,
                self.padSize,
                self.paddedFieldSize,
                self.timeDelta,
                rand_vals,
                order, 
                asyncRadius,
            )
        )

    def Step(self):
        orders = self.orders
        if self.asyncMode == ASYNC_RANDOM_MODE:
            random.shuffle(orders)
        asyncRadius = self.asyncRadius
        for order in orders:
            self.Convolute(order, asyncRadius)
            self.RunGrowth(order, asyncRadius)
            self.RunApply(order, asyncRadius)
            self.SetPadding()

    def StepTimeUnit(self):
        for _ in range(self.stepsPerTimeUnit):
            self.Step()

    def GetState(self):
        return cp.asnumpy(self.img[
            self.padSize:self.paddedFieldSize - self.padSize, 
            self.padSize:self.paddedFieldSize - self.padSize, 
            self.padSize:self.paddedFieldSize - self.padSize
        ])
    
    def GetPaddedState(self):
        return cp.asnumpy(self.img)
    
    def GetStates(self, generations):
        states = [self.GetState()]
        for _ in tqdm(range(generations)):
            self.StepTimeUnit()
            states.append(self.GetState())
        return states

    def __init__(self, startCondition, kernel, growthFunction, applyFunction, stepsPerTimeUnit=1, asyncMode=SYNC_MODE):
        if not len(startCondition.shape) == 3:
            raise ValueError("startCondition is not a 3D array")
        if not (startCondition.shape[0] == startCondition.shape[1] and startCondition.shape[1] == startCondition.shape[2]):
            raise ValueError("startCondition does not have equal sizes")
        if not startCondition.dtype == cp.float32:
            raise ValueError("startCondition is not of type float32")
        if not len(kernel.shape) == 3:
            raise ValueError("kernel is not a 3D array")
        if not (kernel.shape[0] == kernel.shape[1] and kernel.shape[1] == kernel.shape[2]):
            raise ValueError("kernel does not have equal sizes")
        if not kernel.dtype == cp.float32:
            raise ValueError("kernel is not of type float32")
        if kernel.shape[0] % 2 == 0:
            raise ValueError("kernel size is not an odd number")
        if type(stepsPerTimeUnit) != int or stepsPerTimeUnit <= 0:
            raise ValueError("stepsPerTimeUnit is not a positive int")
        
        self.fieldSize = startCondition.shape[0]
        self.kernelSize = kernel.shape[0]
        self.padSize = self.kernelSize // 2
        self.paddedFieldSize = self.fieldSize + 2 * self.padSize

        self.stepsPerTimeUnit = stepsPerTimeUnit
        self.timeDelta = 1.0 / stepsPerTimeUnit

        self.img = cp.zeros((self.paddedFieldSize, self.paddedFieldSize, self.paddedFieldSize), dtype=cp.float32)
        self.img[
            self.padSize:self.paddedFieldSize - self.padSize, 
            self.padSize:self.paddedFieldSize - self.padSize, 
            self.padSize:self.paddedFieldSize - self.padSize
        ] = startCondition
        self.SetPadding()
        self.kernel = kernel
        self.growth = cp.zeros((self.fieldSize, self.fieldSize, self.fieldSize), dtype=cp.float32)

        self.growthFunction = growthFunction
        self.applyFunction = applyFunction

        gridSize = math.ceil(self.fieldSize / THREADS_PER_BLOCK)
        self.gridDim = (gridSize, gridSize, gridSize)
        self.blockDim = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)

        self.asyncMode = asyncMode
        self.asyncRadius = 1 if asyncMode == SYNC_MODE else self.kernelSize
        self.orders = range(self.asyncRadius * self.asyncRadius * self.asyncRadius)
