from functools import reduce
import numpy as np

class dIMF:
    def __dMatrixOfFisher(self, n, s, H, dH, R, cObject, xatk, dxatk):
        delta_M = np.array([[0., 0.], [0., 0.]])
        coeff_dxa_tk_plus_one = np.dot(dxatk, xatk.transpose()) + np.dot(xatk, dxatk.transpose())
        for i in range(2):
            for j in range(2):
                A0 = reduce(np.dot, [dH[i], cObject.CountCi(I=0, n=n, s=s),
                                     coeff_dxa_tk_plus_one,
                                     cObject.CountCi(I=0, n=n, s=s).transpose(), dH[j].transpose(), pow(R, -1)])

                A1 = reduce(np.dot, [dH[i], cObject.CountCi(I=0, n=n, s=s),
                                     coeff_dxa_tk_plus_one,
                                     cObject.CountCi(I=j + 1, n=n, s=s).transpose(), H.transpose(), pow(R, -1)])

                A2 = reduce(np.dot, [H, cObject.CountCi(I=i + 1, n=n, s=s),
                                     coeff_dxa_tk_plus_one,
                                     cObject.CountCi(I=0, n=n, s=s).transpose(), dH[j].transpose(), pow(R, -1)])

                A3 = reduce(np.dot, [H, cObject.CountCi(I=i + 1, n=n, s=s),
                                     coeff_dxa_tk_plus_one,
                                     cObject.CountCi(I=j + 1, n=n, s=s).transpose(), H.transpose(), pow(R, -1)])

                delta_M[i][j] = A0 + A1 + A2 + A3
        return delta_M

    def MaindIMF(self, params, paramObj):
        n = params['n']
        t = params['t']
        s = params['s']
        N = params['N']
        r = params['r']
        H = params['H']
        dH = params['dH']
        R = params['R']

        eMatrix = paramObj["eMatrix"]
        iMatrix = paramObj["iMatrix"]
        xAObject = paramObj["xAObject"]
        dxAObject = paramObj["dxAObject"]
        dAtkObject = paramObj["dAtkObject"]
        FaObject = paramObj["FaObject"]
        AtkObject = paramObj["AtkObject"]
        cObject = paramObj["cObject"]
        dmFisher = np.zeros((N, r, 2, 2))
        datk, dxatk = [], np.zeros((N, r, N, 3 * n, 1))  # shape: N, alpha, betta
        for k in range(N):
            if k == 0:
                xatk = xAObject.StartCountXatk(k, eMatrix, iMatrix)
                for betta in range(N):
                    for alpha in range(r):
                        datk.append(
                            dAtkObject.CountAtk(t=[t[0], t[1]], n=n, k=k, betta=betta, alpha=alpha, eMatrix=eMatrix,
                                                iMatrix=iMatrix))
                        dxatk[k][alpha][betta] = dxAObject.StartCountXatk(datk[alpha])
                        dmFisher[k][alpha] += self.__dMatrixOfFisher(n, s, H, dH, R, cObject, xatk, dxatk[k][alpha][betta])
                    datk.clear()
                continue

            fatk = FaObject.CountFatk(t=[t[k], t[k + 1]], eMatrix=eMatrix)
            atk = AtkObject.CountAtk(t=[t[k], t[k + 1]], index=k, eMatrix=eMatrix, iMatrix=iMatrix)
            xatk = xAObject.ContinueCountXatk(fatk, atk, xatk)

            for betta in range(N):
                for alpha in range(r):
                    datk.append(
                        dAtkObject.CountAtk(t=[t[k], t[k + 1]], n=n, k=k, betta=betta, alpha=alpha, eMatrix=eMatrix,
                                            iMatrix=iMatrix))
                    dxatk[k][alpha][betta] = (
                        dxAObject.ContinueCountXatk(fatk, datk[alpha], dxatk[k - 1][alpha][betta]))
                    dmFisher[k][alpha] += self.__dMatrixOfFisher(n, s, H, dH, R, cObject, xatk, dxatk[k][alpha][betta])
                datk.clear()

        return dmFisher



