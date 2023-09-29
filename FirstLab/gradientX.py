import numpy as np

class gradientX:
    def __init__(self, n, N, tetta):
        self.n = n
        self.N = N
        self.tetta = tetta

    # return F, Psi, H, x_t0
    def initVariables(self, mode):
        if mode == 0:
            self.F = np.array([[-0.8, 1.0], [self.tetta[0][0], 0]])
            self.Psi = np.array([[self.tetta[1][0]], [1.0]])
            self.H = np.array([[1.0, 0]])
            self.R = np.array([[0.1]])
            self.x0 = np.zeros((self.n, 1))
            self.u = np.ones((self.N, 1))
            return self.F, self.Psi, self.H, self.R, self.x0, self.u
        if mode == 1:
            self.F = np.array([[0]])
            self.Psi = np.array([[self.tetta[0][0], self.tetta[1][0]]])
            self.H = np.array([[1.0]])
            self.R = np.array([[0.3]])
            self.x0 = np.zeros((self.n, 1))
            self.u = np.array([[[2.], [1.]], [[1.], [2.]]])
            return self.F, self.Psi, self.H, self.R, self.x0, self.u

    def getValues(self):
        return self.F, self.Psi, self.H, self.R, self.x0, self.u

    def dxdt(self, xi, tk):
        massive = [0, 0]
        xDot = np.dot(self.F, (np.array(xi)).reshape((2, 1))) + np.dot(self.Psi, self.u[0][0])
        for i in range(len(xDot)):
            massive[i] = xDot[i][0]
        return massive