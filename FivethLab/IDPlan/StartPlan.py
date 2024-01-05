import numpy as np
import IIMF.IMF as IIMF

class StartPlan:
    def __GetU(self, q, N):
        U = []
        p = []
        for i in range(q):
            U.append(np.array([[[float(np.random.uniform(0.1, 10.)) for stepi in range(1)]] for stepj in range(N)]))
            p.append(1.0 / q)
        Ksik = {"U": U, "p": p}
        return Ksik
    def __MatrixCount(self, paramVar, Ksik, paramObj):
        iMatrix = paramObj["iMatrix"]
        imfObject = IIMF.IMF()

        U = Ksik["U"]
        p = Ksik["p"]
        q = paramVar["q"]
        matrix = np.zeros((2, 2))

        for i in range(q):
            iMatrix.u = U[i]
            paramObj["iMatrix"] = iMatrix
            matrix += np.dot(p[i], imfObject.MainIMF(paramVar, paramObj))
        return matrix
    def MainCountPlan(self, paramVar, paramObj):
        q = paramVar["q"]
        N = paramVar["N"]
        Ksik = self.__GetU(q, N)
        matrix = self.__MatrixCount(paramVar, Ksik, paramObj)
        return Ksik, matrix

