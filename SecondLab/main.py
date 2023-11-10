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
    def Variables(self, tetta, N, modeName, modeTest):
        # return: F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u
        if modeName == "IMF" and modeTest == 2:
            self.F = np.array([-0.8, 1., tetta[0], 0.])
            self.dF = [np.array([0., 0., 1., 0.]), np.array([0., 0., 0., 0.])]
            self.Psi = np.array([tetta[1], 1.])
            self.dPsi = [np.array([0., 0.]), np.array([1., 0.])]
            self.H = np.array([1., 0.])
            self.dH = [np.array([0., 0.]), np.array([0., 0.])]
            self.R = 0.1
            self.dR = [0., 0.]
            self.x0 = np.array([[0.], [0.]])
            self.dx0 = [np.array([[0.], [0.]]), np.array([[0.], [0.]])]
            self.u = np.array([[1.] for i in range(N)])
            return self.F, self.dF, self.Psi, self.dPsi, self.H, self.dH, self.R, self.dR, self.x0, self.dx0, self.u
        elif modeName == "IMF" and modeTest == 1:
            self.F = np.array([[0.]])
            self.dF = [np.array([[0.]]), np.array([[0.]])]
            self.Psi = np.array([[tetta[0], tetta[1]]])
            self.dPsi = [np.array([[1., 0.]]), np.array([[0., 1.]])]
            self.H = np.array([1.])
            self.dH = [np.array([0.]), np.array([0.])]
            self.R = 0.3
            self.dR = [0., 0.]
            self.x0 = np.zeros((1, 1))
            self.dx0 = [np.zeros((1, 1)) for i in range(2)]
            self.u = np.array([[[1.], [1.]] for i in range(N + 1)])
            return self.F, self.dF, self.Psi, self.dPsi, self.H, self.dH, self.R, self.dR, self.x0, self.dx0, self.u

class TransisionMatrix:
    rP = [0, 0, 0, 0]
    @staticmethod
    def CountMatrix(t0, t1, F, rParam):
        a = (F).reshape((rParam[0], rParam[1]))
        eMatrix = linalg.expm(np.array(F).reshape(rParam[0], rParam[1]) * (t1 - t0))
        return eMatrix
    @staticmethod
    def CountMatrixdF(t0, t1, F, dF, rParam):
        eMatrix = np.dot(linalg.expm(np.array(F).reshape(rParam[0], rParam[1]) * (t1 - t0)),
                              np.array(dF).reshape(rParam[0], rParam[1]) * (t1 - t0))
        return eMatrix
class IntegrateMethods:
    u = [[]]
    @staticmethod
    def A0(x, t1, F, Psi, u, index, eMatrix):
        a0 = reduce(np.dot, [eMatrix.CountMatrix(x, t1, F, eMatrix.rP),
                    np.array(Psi).reshape(eMatrix.rP[2], eMatrix.rP[3]), u[index]])
        return a0
    @staticmethod
    def A1(x, t1, F, dF, Psi, dPsi, u, index, eMatrix):

        a1 = reduce(np.dot, [eMatrix.CountMatrixdF(x, t1, F, dF, eMatrix.rP),
                             np.array(Psi).reshape(eMatrix.rP[2], eMatrix.rP[3]), u[index]]) + \
             reduce(np.dot, [eMatrix.CountMatrix(x, t1, F, eMatrix.rP),
                             np.array(dPsi).reshape(eMatrix.rP[2], eMatrix.rP[3]), u[index]])

        return a1
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
    def Soldering(a1, a2, a3):
        result = []
        for element in a1:
            result.append(element)
        for element in a2:
            result.append(element)
        for element in a3:
            result.append(element)
        return np.array(result)
    def StartCountXatk(self, index, eMatrix, iMartix):
        u = iMartix.u
        a0 = integrate.quad_vec(iMartix.A0, self.t[0], self.t[1],
                                args=(self.t[1], self.F, self.Psi, u, index, eMatrix))[0]

        a1 = integrate.quad_vec(iMartix.A1, self.t[0], self.t[1],
                                args=(self.t[1], (self.F), (self.dF[0]),
                                self.Psi, self.dPsi[0], u, index, eMatrix))[0]

        a2 = integrate.quad_vec(iMartix.A1, self.t[0], self.t[1],
                                args=(self.t[1], self.F, self.dF[1],
                                self.Psi, self.dPsi[1], u, index, eMatrix))[0]
        xat0 = self.Soldering(a0, a1, a2)
        return xat0
    def ContinueCountXatk(self, fatk, atk, xatk):
        return np.dot(fatk, xatk) + atk
class Fatk:
    def __init__(self, F, dF):
        self.F = F
        self.dF = dF
    def CountFatk(self, t, eMatrix):
        O = np.zeros((eMatrix.rP[0], eMatrix.rP[1]))
        a1 = np.hstack((eMatrix.CountMatrix(t[0], t[1], self.F, eMatrix.rP), O, O))

        a2 = np.hstack((eMatrix.CountMatrixdF(t[0], t[1], self.F, self.dF[0], eMatrix.rP),
                        eMatrix.CountMatrix(t[0], t[1], self.F, eMatrix.rP), O))

        a3 = np.hstack((eMatrix.CountMatrixdF(t[0], t[1], self.F, self.dF[1], eMatrix.rP), O,
                        eMatrix.CountMatrix(t[0], t[1], self.F, eMatrix.rP)))
        self.faMatrix = np.vstack((a1, a2, a3))
        return self.faMatrix
    def ReturnFatk(self):
        return self.faMatrix

class Atk:
    def __init__(self, F, dF, Psi, dPsi, x0, dx0):
        self.F = F
        self.dF = dF
        self.Psi = Psi
        self.dPsi = dPsi
        self.x0 = x0
        self.dx0 = dx0
    def CountAtk(self, t, index, eMatrix, iMatrix):
        u = iMatrix.u
        a0 = integrate.quad_vec(iMatrix.A0, t[0], t[1], args=(t[1], self.F, self.Psi, u, index, eMatrix))[0]

        a1 = integrate.quad_vec(iMatrix.A1, t[0], t[1],
                            args=(t[1], self.F, self.dF[0], self.Psi, self.dPsi[0], u, index, eMatrix))[0]

        a2 = integrate.quad_vec(iMatrix.A1, t[0], t[1],
                                args=(t[1], self.F, self.dF[1],self.Psi, self.dPsi[1], u, index, eMatrix))[0]

        self.atk = Xatk.Soldering(a0, a1, a2)
        return self.atk
    def ReturnAtk(self):
        return self.atk

def MatrixOfFisher(n ,s, H, dH, R, mFisher, cObject, xatk):
    delta_M = np.array([[0., 0.], [0., 0.]])
    for i in range(2):
        for j in range(2):
            A0 = reduce(np.dot, [dH[i], cObject.CountCi(I=0, n=n, s=s),
                                 xatk, xatk.transpose(),
                                 cObject.CountCi(I=0, n=n, s=s).transpose(), dH[j].transpose(), pow(R, -1)])

            A1 = reduce(np.dot, [dH[i], cObject.CountCi(I=0, n=n, s=s),
                                 xatk, xatk.transpose(),
                                 cObject.CountCi(I=j + 1, n=n, s=s).transpose(), H.transpose(), pow(R, -1)])

            A2 = reduce(np.dot, [H, cObject.CountCi(I=i + 1, n=n, s=s),
                                 xatk, xatk.transpose(),
                                 cObject.CountCi(I=0, n=n, s=s).transpose(), dH[j].transpose(), pow(R, -1)])

            A3 = reduce(np.dot, [H, cObject.CountCi(I=i + 1, n=n, s=s),
                                 xatk, xatk.transpose(),
                                 cObject.CountCi(I=j + 1, n=n, s=s).transpose(), H.transpose(), pow(R, -1)])

            delta_M[i][j] = A0 + A1 + A2 + A3
    mFisher += delta_M
    return mFisher

def main():
    # Определение переменных
    m = q = v = nu = 1
    n = 2 # Размерность вектора х0
    s = 2 # Количество производных по тетта
    N = 50 # Число испытаний
    tetta_true = np.array([-1.5, 1.0])
    tetta_false = np.array([-1, 1.5])
    t = []

    if n == 1:
        count = 0
        params = [1, 1, 1, 2] # Reshape params for test
        for i in range(N + 1):
            if i == 0:
                t.append(count)
            else:
                count += 0.5
                t.append(count)
    else:
        params = [2, 2, 2, 1] # F (2, 2) and Psi (2, 1)
        t = np.arange(N + 1)

    varObject = IMFVariablesSaver()

    # Choose Test or NoTest
    F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u = varObject.Variables(tetta_true, N, "IMF", modeTest=n)

    eMatrix = TransisionMatrix()
    eMatrix.rP = params
    iMatrix = IntegrateMethods()
    iMatrix.u = u
    xAObject = Xatk(F, dF, Psi, dPsi, t=[t[0], t[1]], x0=x0, dx0=dx0)
    FaObject = Fatk(F, dF)
    AtkObject = Atk(F, dF, Psi, dPsi, x0=x0, dx0=dx0)
    cObject = Ci()

    mFisher = np.array([[0., 0.], [0., 0.]])

    for k in range(N):
        if k == 0:
            xatk = xAObject.StartCountXatk(k, eMatrix, iMatrix)
            mFisher = MatrixOfFisher(n, s, H, dH, R, mFisher, cObject, xatk)
            continue

        fatk = FaObject.CountFatk(t=[t[k], t[k + 1]], eMatrix=eMatrix)
        atk = AtkObject.CountAtk(t=[t[k], t[k + 1]], index=k, eMatrix=eMatrix, iMatrix=iMatrix)
        xatk = xAObject.ContinueCountXatk(fatk, atk, xatk)
        mFisher = MatrixOfFisher(n, s, H, dH, R, mFisher, cObject, xatk)
    print(mFisher)
    a = 0

main()