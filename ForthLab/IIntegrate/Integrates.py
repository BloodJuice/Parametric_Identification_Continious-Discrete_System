from scipy import linalg
from scipy import integrate
import numpy as np
from functools import reduce

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
    u = np.array([[]])
    @staticmethod
    def A0(x, t1, F, Psi, u, index, eMatrix):
        b0 = np.array(Psi).reshape(eMatrix.rP[2], eMatrix.rP[3])
        b_0 = u[index]
        b1 = np.dot(np.array(Psi).reshape(eMatrix.rP[2], eMatrix.rP[3]), u[index])
        b2 = eMatrix.CountMatrix(x, t1, F, eMatrix.rP)
        b3 = np.dot(b2, b0)
        b4 = np.dot(b3, u[index])
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
        u = np.array(iMartix.u)
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

class dXatk:
    def __init__(self, F, dF, Psi, dPsi, t, x0, dx0):
        self.F = F
        self.dF = dF
        self.Psi = Psi
        self.dPsi = dPsi
        self.x0 = x0
        self.dx0 = dx0
        self.t = t

    def StartCountXatk(self, datk):
        dXt0du = datk
        return dXt0du
    def ContinueCountXatk(self, fatk, datk, dxatk):
        return np.dot(fatk, dxatk) + datk

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

        self.atk = np.vstack((a0, a1, a2))
        return self.atk

    def ReturnAtk(self):
        return self.atk

class dAtk:
    def __init__(self, F, dF, Psi, dPsi, x0, dx0, N):
        self.F = F
        self.dF = dF
        self.Psi = Psi
        self.dPsi = dPsi
        self.x0 = x0
        self.dx0 = dx0
        self.N = N
    def CountAtk(self, t, n, k, betta, alpha, eMatrix, iMatrix):
        u, index = [1.], 0

        if (betta == k) and (n == 1):
            dudua = np.zeros((self.N, 1))
            dudua[alpha] = 1.
        elif (betta != k) and (n == 1):
            dudua = np.zeros((self.N, 1))
        elif (betta == k) and (n == 2):
            dudua = 1.
        elif (betta != k) and (n == 2):
            dudua = 0.


        a0 = np.dot(integrate.quad_vec(iMatrix.A0, t[0], t[1], args=(t[1], self.F, self.Psi, u, index, eMatrix))[0], dudua)

        a1 = np.dot(integrate.quad_vec(iMatrix.A1, t[0], t[1],
                            args=(t[1], self.F, self.dF[0], self.Psi, self.dPsi[0], u, index, eMatrix))[0], dudua)

        a2 = np.dot(integrate.quad_vec(iMatrix.A1, t[0], t[1],
                                args=(t[1], self.F, self.dF[1],self.Psi, self.dPsi[1], u, index, eMatrix))[0], dudua)

        self.datk = np.vstack((a0, a1, a2))
        return self.datk

    def ReturnAtk(self):
        return self.datk
