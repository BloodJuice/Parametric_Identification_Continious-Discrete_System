from functools import reduce
import numpy as np
from scipy import linalg
from scipy import integrate

class Ci:
    def CountCi(self, I, n, s):
        self.matrix_ci = np.zeros((n, n * (s + 1)))
        for _ in range(n):
            self.matrix_ci[_][I * n + _] = 1
        return self.matrix_ci
    def ReturnCi(self):
        return self.matrix_ci

# return F, Psi, H, R, x0, u
class IMFVariablesSaver:
    # modes: IMF, IMF_test, dIMF, dIMF_test
    def Variables(self, tetta, mode, N):
        # return: F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u
        if mode == "IMF":
            self.F = np.array([[-0.8, 1.], [tetta[0], 0.]])
            self.dF = [np.array([[0., 0.], [1., 0.]]), np.array([[0., 0.], [0., 0.]])]
            self.Psi = np.array([[tetta[1]], [1.]])
            self.dPsi = [np.array([[0.], [0.]]), np.array([[1.], [0.]])]
            self.H = np.array([1., 0.])
            self.dH = [np.array([0., 0.]), np.array([0., 0.])]
            self.R = 0.1
            self.dR = [0., 0.]
            self.x0 = np.array([[0.], [0.]])
            self.dx0 = [np.array([[0.], [0.]]), np.array([[0.], [0.]])]
            self.u = np.array([[1.] for i in range(N)])
            return self.F, self.dF, self.Psi, self.dPsi, self.H, self.dH, self.R, self.dR, self.x0, self.dx0, self.u

class TransisionMatrix:
    def CountMatrix(self, t0, t1, F):
        self.eMatrix = linalg.expm(F * (t1 - t0))
        return self.eMatrix
    def CountMatrixdF(self, t0, t1, F, dF):
        self.eMatrix = np.dot(linalg.expm(F * (t1 - t0)), dF * (t1 - t0))
        return self.eMatrix
    def ReturnEMatrix(self):
        return self.eMatrix
class Xatk:
    def __init__(self, F, dF, Psi, dPsi, t, x0, dx0):
        self.F = F
        self.dF = dF
        self.Psi = Psi
        self.dPsi = dPsi
        self.x0 = x0
        self.dx0 = dx0
        self.t = t

    @staticmethod
    def LineElement2D(M):
        result = []
        for line in M:
            for element in line:
                result.append(element)
        return result
    @staticmethod
    def A0(x, t1, F, Psi):
        eMatrix = TransisionMatrix()
        a0 = np.dot(eMatrix.CountMatrix(x, t1, np.array(F).reshape(2, 2)), np.array(Psi).reshape(2, 1))
        return a0

    @staticmethod
    def A1(x, t1, F, dF, Psi, dPsi):
        eMatrix = TransisionMatrix()
        a1 = np.dot(eMatrix.CountMatrixdF(x, t1, np.array(F).reshape(2, 2), np.array(dF).reshape(2, 2)), np.array(Psi).reshape(2, 1)) + \
             np.dot(eMatrix.CountMatrix(x, t1, np.array(F).reshape(2, 2)), np.array(dPsi).reshape(2, 1))
        return a1

    @staticmethod
    def Soldering(a1, a2, a3):
        result = []
        for element in a1:
            result.append(element)
        for element in a2:
            result.append(element)
        for element in a3:
            result.append(element)
        return np.array(result)
    def StartCountXatk(self):
        eMatrix = TransisionMatrix()
        a0 = np.dot(eMatrix.CountMatrix(self.t[0], self.t[1], self.F), self.x0) + \
             integrate.quad_vec(self.A0, self.t[0], self.t[1], args=(self.t[1], self.LineElement2D(self.F), self.LineElement2D(self.Psi)))[0]

        a1 = integrate.quad_vec(self.A1, self.t[0], self.t[1], args=(self.t[1], self.LineElement2D(self.F), self.LineElement2D(self.dF[0]),
                self.LineElement2D(self.Psi), self.LineElement2D(self.dPsi[0])))[0]

        a2 = integrate.quad_vec(self.A1, self.t[0], self.t[1], args=(self.t[1], self.LineElement2D(self.F), self.LineElement2D(self.dF[1]),
                self.LineElement2D(self.Psi), self.LineElement2D(self.dPsi[1])))[0]
        xat0 = self.Soldering(a0, a1, a2)
        return xat0


def main():
    # Определение переменных
    m = q = v = nu = 1

    n = 2 # Размерность вектора х0
    s = 2 # Количество производных по тетта
    N = 3 # Число испытаний

    count = 0

    delta = 0.001

    tetta_true = np.array([-1.5, 1.0])
    tetta_false = np.array([-1, 1])

    varObject = IMFVariablesSaver()
    F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u = varObject.Variables(tetta_true, "IMF", N)

    xAObject = Xatk(F, dF, Psi, dPsi, t=[0., 1.], x0=x0, dx0=dx0)
    xAObject.StartCountXatk()


main()