import numpy as np
from scipy.integrate import odeint
import InitialForFirst
class Non_gradients():

    def setTetta(self, tetta):
        self.tetta = tetta

    def Xi(self, tetta, params):
        N = params["N"]
        ki = params["ki"]
        q = params["q"]
        ytk = params["y"]
        s = params["s"]
        initObj = InitialForFirst.InitialForFirst()

        valuesAndGradX = params["gradX"]

        F, Psi, H, R, x0, dF, dPsi, dH, dR, dx0 = initObj.MainVariables(tetta=tetta, mode="Main", params={"N": N, "s": s})

        x0 = np.array(x0).reshape(2, 1)
        U = params["U"]

        xt = [[(np.array([0., 0.])).reshape(2, 1) for j in range(N)] for i in range(q)]
        tk = np.arange(N)
        xi = N * params['m'] * params['v'] * np.log(2 * np.pi) + N * params['v'] * np.log(R)

        for k in range(N - 1):
            delta = 0.
            for i in range(q):
                if k == 0:
                    xt[i][k] = x0

                # Поиск производной dxdt:
                tNow = [tk[k], tk[k + 1]]
                dxdtk_One = odeint(valuesAndGradX.dxdt, xt[i][k].reshape(2,), tNow, args=(F, Psi, H, R, x0, U[i][k]))[1]
                xt[i][k + 1] = (np.array(dxdtk_One)).reshape(2, 1)
                for j in range(int(ki[i])):
                    epstk = ytk[i][0][k + 1] - np.dot(H, xt[i][k + 1])
                    delta += np.dot(np.dot(epstk.transpose(), pow(R, -1)), epstk)
            xi += delta
        return xi[0][0] / 2.0