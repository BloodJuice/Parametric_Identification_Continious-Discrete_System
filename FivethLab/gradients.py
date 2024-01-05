import numpy as np
import non_gradients
from scipy.integrate import odeint
import InitialForFirst

class Gradients():

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
    def dxdt(self, xi, tk, F, Psi, H, R, x0, U):
        massive = [0., 0.]
        xDot = np.dot(F, (np.array(xi)).reshape((2, 1))) + np.dot(Psi, U)
        for i in range(len(xDot)):
            massive[i] = xDot[i][0]
        return massive

    def dxdtAlpha(self, dxi, tk, alpha, x1, x2, F, dF, dPsi, U):
        massive = [0., 0.]
        xi = [x1, x2]
        a1 = np.dot(dF[alpha], np.array(xi).reshape(2, 1))
        a2 = np.dot(F, (np.array(dxi)).reshape((2, 1)))
        a3 = np.dot(dPsi[alpha], U[0][0])
        dxDot = a1 + a2 + a3
        for i in range(len(dxDot)):
            massive[i] = dxDot[i][0]
        return massive


    def dXi(self, tetta, params):
        ###############___0.5____###################################
        v = params["v"]
        ki = params["ki"]
        m = params['m']
        R = params["R"]
        s = params["s"]
        N = params["N"]
        tk = np.arange(N)
        tNow = []
        y = params["y"]

        gradX = params["gradX"]
        initObj = InitialForFirst.InitialForFirst()
        F, Psi, H, R, xt0, dF, dPsi, dH, dR, dxt0 = initObj.MainVariables(tetta=tetta, mode="Main",
                                                                        params={"N": N, "s": s})

        u = params["U"]

        # 2
        gradient = (np.array([v / 2. * N * 1. / R * dR[alpha] for alpha in range(s)])).reshape(s, 1)

        q = params["q"]
        # Point 5
        xt = [[(np.array([0., 0.])).reshape(2, 1) for k in range(N)] for i in range(q)]
        dxtk = [[[(np.array([0., 0.])).reshape(2, 1) for k in range(N)] for i in range(q)] for alpha in range(s)]


        deps = [[np.array([[([0.]) for k in range(N)] for j in range(int(ki[i]))]) for i in range(q)] for alpha in range(s)]
        eps = [np.array([[([0.]) for k in range(N)] for j in range(int(ki[i]))]) for i in range(q)]
        for k in range(N - 1):
            delta = np.array([0. for alpa in range(s)])  # Инициализация треугольничка
            for i in range(0, q):
                # Point 5
                if k == 0:
                    xt[i][k] = xt0
                    for alpha in range(s):
                        dxtk[alpha][i][k] = dxt0[alpha]
                # Point 6
                tNow = [tk[k], tk[k + 1]]
                # xi, tk, F, Psi, H, R, x0, U
                dxdtk_One = (odeint(self.dxdt, xt[i][k].reshape(2,), tNow, args=(F, Psi, H, R, xt0, u[i][k]))[1])
                xt[i][k + 1] = np.array(dxdtk_One).reshape(2, 1)

                # Point 8
                for j in range(int(ki[i])):
                    for alpha in range(s):
                        # dxtk[alpha][i][k + 1] = np.dot(dF[alpha], xt[i][k]) + np.dot(F, dxtk[alpha][i][k]) \
                        #                     + np.dot(dPsi[alpha], u[i][k])

                        c = xt[i][k].reshape(2, )
                        dxtk[alpha][i][k + 1] = ((np.array(odeint(gradX.dxdtAlpha, dxtk[alpha][i][k].reshape(2, ), tNow,
                                                    args=(alpha, c[0], c[1], F, dF, dPsi, u[i][k]))[1])).reshape(2, 1))

                        deps[alpha][i][j][k + 1] = (-1) * np.dot(dH[alpha], xt[i][k + 1]) - np.dot(H,
                                                                                              dxtk[alpha][i][k + 1])

                # Point 9
                    eps[i][j][k + 1] = y[i][j][k + 1] - np.dot(H, xt[i][k + 1])
                    for alpha in range(s):
                        delta[alpha] += np.dot(np.dot(deps[alpha][i][j][k + 1].transpose(), pow(R, -1)), eps[i][j][k + 1]) - \
                                (0.5) * np.dot(np.dot(eps[i][j][k + 1].transpose(), pow(R, -1)), dR[alpha]) * \
                                                                               np.dot(pow(R, -1), eps[i][j][k + 1])
            for alpha in range(s):
                gradient[alpha] += delta[alpha]

                # if ki[i] + 1 < N:
                #     xt[i][ki + 1] = xPlusOne
                #     for alpha in range(s):
                #         dxtk[alpha][i][ki[i] + 1] = dxtk[alpha][i][ki[i]]
                # else:
                #     xt[i][ki] = xPlusOne
                #     for alpha in range(s):
                #         dxtk[alpha][i][ki[i]] = dxtk[alpha][i][ki[i]]
        return gradient
