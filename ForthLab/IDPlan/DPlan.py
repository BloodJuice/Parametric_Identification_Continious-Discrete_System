import numpy as np
from scipy.optimize import minimize, Bounds
import IIMF.IMF as IIMF
import IIMF.dIMF as dIIMF

class DPlan:
    def MainDPlan(self, Ksik, paramVar, paramObj, mode):
        N = paramVar["N"]
        q = paramVar["q"]

        U = Ksik["U"]
        p = Ksik["p"]
        matrix = np.linalg.inv(self.__MatrixCount(U, p, paramVar, paramObj))
        U = self.__LineU(Ksik["U"])

        paramVar["matrix"] = matrix

        if mode == "dUXMKsik":
            result = minimize(self.__XMKsikForUi, U, args=(p, paramVar, paramObj), method='SLSQP',
                              jac=self.__dXMKsikForUi, bounds=Bounds([0.]*N*q, [10.]*q*N))
            KsikNew = self.__ReturnMatrixU(result.__getitem__("x"), q, N)
            return KsikNew
        elif mode == "dPXMKsik":

            start_pos = np.ones(q) * 0.

            # Says one minus the sum of all variables must be zero

            cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
            # Required to have non negative values
            bnds = tuple((0, 1) for x in start_pos)

            result = minimize(self.__XMKsikForPi, p, args=(U, paramVar, paramObj), method='SLSQP',
                              jac=self.__dXMKsikForPi, bounds=bnds, constraints=cons)
            pi = result.__getitem__("x")
            return pi

    def __LineU(self, U):
        result = []
        for line in U:
            for element in line:
                result.append(element[0][0])
        return result

    def __ReturnMatrixU(self, U, q, N):
        result = []
        globalcount = 0
        for i in range(q):
            line = []
            for j in range(N):
                line.append([[U[globalcount]]])
                globalcount += 1
            result.append(np.array(line.copy()))
        return result

    def __MatrixCount(self, Ksik, p, paramVar, paramObj):
        iMatrix = paramObj["iMatrix"]
        imfObject = IIMF.IMF()

        U = Ksik
        q = paramVar["q"]
        matrix = np.zeros((2, 2))

        for i in range(q):
            iMatrix.u = U[i]
            paramObj["iMatrix"] = iMatrix
            matrix += np.dot(p[i], imfObject.MainIMF(paramVar, paramObj))
        return matrix

    #######################____SecondPointStart____#######################
    def __dXMKsikForUi(self, U, p, paramVar, paramObj):
        N = paramVar["N"]
        q = paramVar["q"]
        iMatrix = paramObj["iMatrix"]
        dimfObj = dIIMF.dIMF()
        result = []

        U = self.__ReturnMatrixU(U, q, N)
        # matrix = np.linalg.inv(self.__MatrixCount(U, p, paramVar, paramObj))
        matrix = paramVar["matrix"]
        for i in range(q):
            iMatrix.u = U[i]
            paramObj["iMatrix"] = iMatrix
            C0 = (np.dot( matrix,    dimfObj.MaindIMF(paramVar, paramObj) )).trace()
            result.append((-1.) * np.dot(p[i], C0))
        return result

    def __XMKsikForUi(self, U, p, paramVar, paramObj):
        N = paramVar["N"]
        q = paramVar["q"]
        U = self.__ReturnMatrixU(U, q, N)
        result = (-1.) * np.log(np.linalg.det(self.__MatrixCount(U, p, paramVar, paramObj)))
        return result

    #######################____SecondPointFinish____#######################

    #######################____ThirdPointStart____#######################
    def __XMKsikForPi(self, p, U, paramVar, paramObj):
        N = paramVar["N"]
        q = paramVar["q"]
        U = self.__ReturnMatrixU(U, q, N)
        result = (-1.) * np.log(np.linalg.det(self.__MatrixCount(U, p, paramVar, paramObj)))
        return result


    def __dXMKsikForPi(self, p, U, paramVar, paramObj):
        N = paramVar["N"]
        q = paramVar["q"]
        iMatrix = paramObj["iMatrix"]
        imfObj = IIMF.IMF()
        result = []

        U = self.__ReturnMatrixU(U, q, N)
        # matrix = np.linalg.inv(self.__MatrixCount(U, p, paramVar, paramObj))
        matrix = paramVar["matrix"]
        for i in range(q):
            iMatrix.u = U[i]
            paramObj["iMatrix"] = iMatrix
            C0 = (np.dot(matrix, imfObj.MainIMF(paramVar, paramObj))).trace()
            result.append((-1.) * C0)

        return result

    #######################____ThirdPointFinish____#######################