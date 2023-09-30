from scipy.integrate import odeint
import numpy as np
import gradients as gr
from scipy.optimize import minimize, Bounds

def minimizeFirst(tettaMin, params):
    ############__2__##########
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    result = minimize(fun=Xi, x0=tettaMin, args=params,  method="SLSQP", bounds=bounds)
    # res.append(minimize(Xi, x_start, method='SLSQP', jac=dXi, bounds=bounds))
    # print("Тетты для нулевого порядка:", result.__getitem__("x"))
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

def Xi(tetta, params):
    N = params["N"]
    ki = params["ki"]
    q = params["q"]
    ytk = params["y"]

    valuesAndGradX = params["valuesAndGradX"]
    valuesAndGradX.setTetta(tetta)
    F, Psi, H, R, x0, u = valuesAndGradX.initVariables(mode=0)
    valuesAndGradX = params["valuesAndGradX"]
    xt = [[(np.array([0., 0.])).reshape(2, 1) for j in range(N)] for i in range(q)]
    tk = np.arange(N)
    Xi = N * params['m'] * params['v'] * np.log(2 * np.pi) + N * params['v'] * np.log(R)

    for k in range(N - 1):
        delta = 0
        for i in range(q):
            if k == 0:
                xt[i][k] = x0

            # Поиск производной dxdt:
            tNow = [tk[k], tk[k + 1]]
            dxdtk_One = odeint(valuesAndGradX.dxdt, xTransform(xt[i][k]), tNow)[1]
            xt[i][k + 1] = (np.array(dxdtk_One)).reshape(2, 1)
            for j in range(ki):
                epstk = ytk[k + 1] - np.dot(H, xt[i][k + 1])
                delta += np.dot(np.dot(epstk.transpose(), pow(R, -1)), epstk)
        Xi += delta
    return Xi[0][0] / 2.0

def y(xtk, params):
    H = params['H']
    R = params['R']
    yEnd = []
    for i in range(len(xtk)):
        yEnd.append((np.dot(H, xtk[i]) + np.random.normal(0, 1) * R))
    return yEnd

def dXi(tetta, params):
    a = 0

def main():
    # Определение переменных
    r = 2  # Количество начальных сигналов, альфа
    n = 2  # Размерность вектора х0
    s = 2  # Количество производных по тетта
    N = 30  # Число испытаний
    ki = 1
    q = 1
    m = v = 1.
    tetta_false = [-2, 0.5]
    tetta_true = [-1.5, 1]
    tk = np.arange(N + 1)
    valuesAndGradX = gr.Gradients(n, N, s=s, tetta=tetta_true)
    F, Psi, H, R, x0, u = valuesAndGradX.initVariables(mode=0)

    dxtdt_start = []
    x0 = [0.0, 0.0]
    for i in range(len(tk) - 1):
        tNow = [tk[i], tk[i + 1]]
        if i == 0:
            dxtdt_start.append((np.array(odeint(valuesAndGradX.dxdt, x0, tNow)[1])).reshape(2, 1))
        else:
            dxtdt_start.append((np.array(odeint(valuesAndGradX.dxdt, xTransform(dxtdt_start[i - 1]), tNow)[1])).reshape(2, 1))


    params = {"N": N, "ki": ki, "q": q, "y": y(dxtdt_start, {"H":H, "R":R}), "valuesAndGradX":valuesAndGradX, "m":m, "v":v}
    print(f'Result is: \n{minimizeFirst(tetta_false, params)}')



    return 0

main()


