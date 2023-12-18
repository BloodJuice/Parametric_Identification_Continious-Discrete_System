from functools import reduce
import numpy as np

class IMF:
    def __MatrixOfFisher(self, n, s, H, dH, R, mFisher, cObject, xatk):
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

    def MainIMF(self, params, paramObj):
        n = params['n']
        t = params['t']
        s = params['s']
        N = params['N']
        H = params['H']
        dH = params['dH']
        R = params['R']
        eMatrix = paramObj["eMatrix"]
        iMatrix = paramObj["iMatrix"]
        xAObject = paramObj["xAObject"]
        FaObject = paramObj["FaObject"]
        AtkObject = paramObj["AtkObject"]
        cObject = paramObj["cObject"]
        mFisher = np.array([[0., 0.], [0., 0.]])

        for k in range(N):
            if k == 0:
                xatk = xAObject.StartCountXatk(k, eMatrix, iMatrix)
                mFisher = self.__MatrixOfFisher(n, s, H, dH, R, mFisher, cObject, xatk)
                continue

            fatk = FaObject.CountFatk(t=[t[k], t[k + 1]], eMatrix=eMatrix)
            atk = AtkObject.CountAtk(t=[t[k], t[k + 1]], index=k, eMatrix=eMatrix, iMatrix=iMatrix)
            xatk = xAObject.ContinueCountXatk(fatk, atk, xatk)
            mFisher = self.__MatrixOfFisher(n, s, H, dH, R, mFisher, cObject, xatk)
        return mFisher