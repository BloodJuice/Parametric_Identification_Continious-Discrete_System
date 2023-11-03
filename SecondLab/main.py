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
            self.F = np.matrix(f'-0.8 1; {tetta[0]} 0')
            self.dF = [np.matrix(f'0 0; 1 0'), np.matrix(f'0 0; 0 0')]
            self.Psi = np.matrix(f'{tetta[1]}; 1')
            self.dPsi = [np.matrix(f'0; 0'), np.matrix(f'1; 0')]
            self.H = np.matrix(f'1 0')
            self.dH = [np.matrix(f'0 0'), np.matrix(f'0 0')]
            self.R = 0.1
            self.dR = [0., 0.]
            self.x0 = np.matrix(f'0; 0')
            self.dx0 = [np.matrix(f'0; 0'), np.matrix(f'0; 0')]
            self.u = np.array([[1.] for i in range(N)])
            return self.F, self.dF, self.Psi, self.dPsi, self.H, self.dH, self.R, self.dR, self.x0, self.dx0, self.u

class TransisionMatrix:
    def CountMatrix(self, t0, t1, F):
        self.eMatrix = linalg.expm(F * (t1 - t0))
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

    def A0(x, t1, F, Psi):
        matrix = TransisionMatrix()
        a0 = matrix.CountMatrix(x, t1, (F).reshape(2, 2))
        return np.dot(a0, (Psi).reshape(2, 1))
    def StartCountXatk(self):
        eMatrix = TransisionMatrix()
        f0 = np.dot(eMatrix.CountMatrix(self.t[0], self.t[1], self.F), self.x0) + \
             integrate.quad(Xatk.A0, self.t[0], self.t[1], args=(self.t[1], (self.F).reshape(4, )[0], (self.Psi).reshape(2,)[0]))
        a1 = 0

def FTransform(F):
    result = []
    for line in F:
        for element in line:
            result.append(element)

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