from scipy.integrate import odeint
import numpy as np
import gradients as gr
import non_gradients as non_gr
import InitialForFirst
from scipy.optimize import minimize, Bounds


def minimizeFirst(tettaMin, params):
    non_gradX = params["non_gradX"]
    gradX = params["gradX"]
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    result = minimize(fun=non_gradX.Xi, x0=tettaMin, args=params, jac=gradX.dXi,  method="SLSQP", bounds=bounds)
    print("Тетты для первого порядка:\n", result)
    return np.array(result.__getitem__("x")). reshape(2, 1)

def xTransform(massive):
    m = len(massive)
    n = len(massive[0])
    result = []
    for i in range(m):
        for j in range(n):
            result.append(massive[i][j])
    return result


def y(xtk, params):
    H = params['H']
    R = params['R']
    yEnd = []
    for i in range(len(xtk)):
        yEnd.append((np.dot(H, xtk[i]) + np.random.normal(0., 1.) * R))
    return yEnd


def first(params):
    # Определение переменных
    r = params["r"]  # Количество начальных сигналов, альфа
    n = params["n"]  # Размерность вектора х0
    s = params["s"]  # Количество производных по тетта
    N = params["N"]  # Число испытаний
    ki = params["ki"]  # Initial number of system start
    q = params["q"]
    m = params["m"]
    v = params["v"]
    U = params["u"]

    tetta_false = params["tetta_false"]
    tetta_true = params["tetta_true"]
    tk = np.arange(N + 1)
    gradX = gr.Gradients()
    non_gradX = non_gr.Non_gradients()
    initObj = InitialForFirst.InitialForFirst()

    F, Psi, H, R, x0, dF, dPsi, dH, dR, dx0 = initObj.MainVariables(tetta_true, "Main", {"N": N, "s": s})
    # paramInit = {"F": F, "Psi": Psi, "H": H, "R": R, "x0": x0, "dF": dF,
    #              "dPsi": dPsi, "dH": dH, "dR": dR, "dx0": dx0, "U": U}

    dxtdt_start = [[(np.array([0., 0.])).reshape(2, 1) for k in range(N)] for i in range(q)]
    x0 = [0.0, 0.0]
    for k in range(N - 1):
        for i in range(q):
            tNow = [tk[i], tk[i + 1]]
            if i == 0:
                a = odeint(gradX.dxdt, dxtdt_start[i][k].reshape(2,), tNow,
                                                    args=(F, Psi, H, R, x0, U[i][k]))
                dxtdt_start[i][k + 1] = ((np.array(odeint(gradX.dxdt, dxtdt_start[i][k].reshape(2,), tNow,
                                                    args=(F, Psi, H, R, x0, U[i][k]))[1])).reshape(2, 1))
            else:
                dxtdt_start[i][k + 1] = ((np.array(odeint(gradX.dxdt,
                                                    dxtdt_start[i][k].reshape(2,), tNow, args=(F, Psi, H, R, x0, U[i][k]))[1])).reshape(2, 1))

        dx = [[[(np.array([0., 0.])).reshape(2, 1) for k in range(N)] for i in range(q)] for alpha in range(s)]

        for i in range(q):
            for alpha in range(s):
                tNow = [tk[i], tk[i + 1]]
                if i == 0:
                    c = dxtdt_start[i][k].reshape(2, )
                    b = odeint(func=gradX.dxdtAlpha, y0=x0, t=tNow, args=(alpha, c[0], c[1], F, dF, dPsi, U[i][k]))
                    dx[alpha][i][k + 1] = ((np.array(odeint(func=gradX.dxdtAlpha, y0=dx[alpha][i][k].reshape(2,), t=tNow,
                                                     args=(alpha, c[0], c[1], F, dF, dPsi, U[i][k]))[1])).reshape(2, 1))
                else:
                    c = dxtdt_start[i][k].reshape(2,)
                    d = odeint(gradX.dxdtAlpha, dx[alpha][i][k].reshape(2,), tNow,
                                                     args=(alpha, c[0], c[1], F, dF, dPsi, U[i][k]))
                    dx[alpha][i][k + 1] = ((np.array(odeint(gradX.dxdtAlpha, dx[alpha][i][k].reshape(2,), tNow,
                                                     args=(alpha, c[0], c[1], F, dF, dPsi, U[i][k]))[1])).reshape(2, 1))


    F, Psi, H, R, x0, dF, dPsi, dH, dR, dx0 = initObj.MainVariables(tetta_false, "Main", {"N": N, "s": s})
    h = y(dxtdt_start, {"H": H, "R": R})
    params = {"N": N, "ki": ki, "q": q, "s": s, "m": m, "v": v, "R": R,
              "U": U, "y": y(dxtdt_start, {"H": H, "R": R}), "non_gradX": non_gradX, "gradX": gradX}
    result = minimizeFirst(tetta_false, params).reshape(2,)
    print(f'Result is: \n{result}')
    return result


