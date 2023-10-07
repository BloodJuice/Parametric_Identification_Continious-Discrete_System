import numpy as np
import non_gradients
from scipy.integrate import odeint

class Gradients(non_gradients.IGradient):
    def __init__(self, n, N, s, tetta):
        self.n = n
        self.N = N
        self.s = s
        self.tetta = tetta

    # return F, Psi, H, x_t0
    def getValues(self, mode):
        # mode == 0 был необходим для первой лабы, для вычисления dxdt
        if mode == 0:
            self.F = np.array([[-0.8, 1.0], [self.tetta[0], 0]])
            self.Psi = np.array([[self.tetta[1]], [1.0]])
            self.H = np.array([[1.0, 0]])
            self.R = 0.1
            self.x0 = np.zeros((self.n, 1))
            self.u = np.ones((self.N, 1))
            return self.F, self.Psi, self.H, self.R, self.x0, self.u
        elif mode == 1:
            self.dF = np.array([np.array([[0]]), np.array([[0]])])
            self.dPsi = np.array([np.array([[1, 0]]), np.array([[0, 1]])])
            self.dH = np.array([np.array([[0]]), np.array([[0]])])
            self.dR = np.array([np.array([[0]]), np.array([[0]])])
            self.dx0 = np.array(np.zeros((self.n, 1)) for i in range(self.s))
            self.du_dua = np.array([[[1], [0]], [[0], [1]]])
            return self.dF, self.dPsi, self.dH, self.dR, self.dx0, self.du_dua
        if mode == 2:
            self.dF = np.array([np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])])
            self.dPsi = np.array([np.array([[0], [0]]), np.array([[1], [0]])])
            self.dH = np.array([np.array([[0, 0]]), np.array([[0, 0]])])
            self.dR = np.array([np.array([[0]]), np.array([[0]])])
            self.dx0 = np.array([np.zeros((self.n, 1)) for i in range(self.s)])
            self.du_dua = 1
            return self.dF, self.dPsi, self.dH, self.dR, self.dx0, self.du_dua

    def setTetta(self, tetta):
        self.tetta = tetta
    def xTransform(self, massive):
        m = len(massive)
        n = len(massive[0])
        result = []
        for i in range(m):
            for j in range(n):
                result.append(massive[i][j])
        return result
    def dxdt(self, xi, tk):
        massive = [0, 0]
        xDot = np.dot(self.F, (np.array(xi)).reshape((2, 1))) + np.dot(self.Psi, self.u[0][0])
        for i in range(len(xDot)):
            massive[i] = xDot[i][0]
        return massive

    def dxdtAlpha(self, dxi, tk, *args):
        massive = [0, 0]
        alpha = args[0]
        xi = [args[1], args[2]]
        dxDot = np.dot(self.dF[alpha], xi) + np.dot(self.F, (np.array(dxi)).reshape((2, 1))) \
                                                                                    + np.dot(self.Psi[alpha], self.u[0][0])
        for i in range(len(dxDot)):
            massive[i] = dxDot[i][0]
        return massive


    def dXi(self, tetta, params):
        ###############___0.5____###################################
        mode = 2
        ytk = params["y"]
        v = params["v"]
        ki = params["ki"]
        m = params['m']
        R = params['R']
        s = params["s"]
        N = params["N"]
        tk = np.arange(N)
        tNow = []

        gradX = params["gradX"]
        self.tetta = tetta
        F, Psi, H, o, xt0, u = gradX.getValues(mode=0)
        dF, dPsi, dH, dR, dxt0, du_dua = gradX.getValues(mode=2)

        # 2
        gradient = (np.array([v / 2. * N * 1. / R * dR[alfa] for alfa in range(s)])).reshape(s, 1)

        q = ki
        # Point 5
        xt = np.array(
            [np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)])
        dxtk = np.array(
            [[np.full(shape=2, fill_value=0, dtype=float).reshape(2, 1) for stepj in range(N)]for alpha in range(s)])

        # dxPlusOne = np.array([0 for alpha in range(s)])
        depsPlusOne = np.array([0 for alpha in range(s)])

        for k in range(N - 1):
            delta = np.array([0 for alpa in range(s)])  # Инициализация треугольничка
            for i in range(0, q):
                # Point 5
                if k == 0:
                    xt[i][k] = xt0
                    for alpha in range(s):
                        dxtk[alpha][i][k] = dxt0[alpha]
                # Point 6
                tNow = [tk[k], tk[k + 1]]
                dxdtk_One = odeint(Gradients.dxdt, Gradients.xTransform(self, massive=xt[i][k]), tNow)[1]
                xt[i][k + 1] = dxdtk_One

                # Point 7
                for alpha in range(s):
                    dxtk[alpha][i][k] = np.dot(dF[alpha], xtk[i][k]) + np.dot(F, dxtk[alpha][i][
                        k]) + np.dot(dPsi[alpha], u[i][k][0])
                    depsPlusOne[alpha] = (-1) * np.dot(dH[alpha], xtk[alpha][k]) - np.dot(H,
                                                                                               dxtk[alpha][i][k])

                for j in range(int(ki)):
                    epsPlusOne = y[i][ki][0] - np.dot(H, xPlusOne)
                    for alpha in range(s):
                        delta[alpha] += np.dot(np.dot(depsPlusOne[alpha].transpose(), pow(R, -1)), epsPlusOne) - \
                                           (0.5) * np.dot(np.dot(depsPlusOne[alpha].transpose(), pow(R, -1)),
                                                          dR[alpha]) * \
                                           np.dot(pow(R, -1), epsPlusOne)
            for alpha in range(s):
                gradient[alpha] += delta[alpha]

            if ki + 1 < N:
                xtk[i][ki + 1] = xPlusOne
                for alpha in range(s):
                    dxtk[alpha][i][ki + 1] = dxtk[alpha][i][ki]
            else:
                xtk[i][ki] = xPlusOne
                for alpha in range(s):
                    dxtk[alpha][i][ki] = dxtk[alpha][i][ki]
        return gradient
