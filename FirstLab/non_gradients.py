import numpy as np
from abc import ABC, abstractmethod


class IGradient(ABC):
    @abstractmethod
    def initVariables(self, mode):
        pass

    @abstractmethod
    def setTetta(self, tetta):
        pass

class Non_gradients(IGradient):
    def __init__(self, n, N, tetta):
        self.n = n
        self.N = N
        self.tetta = tetta

    def initVariables(self, mode):
        if mode == 0:
            self.F = np.array([[-0.8, 1.0], [self.tetta[0], 0]])
            self.Psi = np.array([[self.tetta[1]], [1.0]])
            self.H = np.array([[1.0, 0]])
            self.R = 0.1
            self.x0 = np.zeros((self.n, 1))
            self.u = np.ones((self.N, 1))
            return self.F, self.Psi, self.H, self.R, self.x0, self.u
        if mode == 1:
            self.F = np.array([[0]])
            self.Psi = np.array([[self.tetta[0], self.tetta[1]]])
            self.H = np.array([[1.0]])
            self.R = 0.3
            self.x0 = np.zeros((self.n, 1))
            self.u = np.array([[[2.], [1.]], [[1.], [2.]]])
            return self.F, self.Psi, self.H, self.R, self.x0, self.u

    def setTetta(self, tetta):
        self.tetta = tetta

    def getValues(self):
        return self.F, self.Psi, self.H, self.R, self.x0, self.u