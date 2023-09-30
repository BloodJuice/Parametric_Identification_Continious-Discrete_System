from scipy.integrate import odeint
import numpy as np
import gradientX as gr
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint

def minimizeFirst(tettaMin, params):
    ############__2__##########
    lim = [-2.0, 0.01, -0.05, 1.5]
    bounds = Bounds([lim[0], lim[1]],  # [min x0, min x1]
                    [lim[2], lim[3]])  # [max x0, max x1]
    result = minimize(fun=Xi, x0=tettaMin, args=params,  method="L-BFGS-B", bounds=bounds)
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
    N = params['N']
    ki = params['ki']
    q = params['q']
    F = params['F']
    Psi = params['Psi']
    H = params['H']
    R = params['R']
    x0 = params['x0']
    u = params['u']
    valuesAndGradX = params["valuesAndGradX"]
    xtNow = [[(np.array([0., 0.])).reshape(2, 1) for j in range(N)] for i in range(q)]
    tk = np.arange(N)
    Xi = N * params['m'] * params['v'] * np.log(2 * np.pi) + N * params['v'] * np.log(R)

    for k in range(N - 1):
        delta = 0
        for i in range(q):
            if k == 0:
                xtNow[i][k] = x0

            # Поиск производной dxdt:
            tNow = [tk[k], tk[k + 1]]
            dxdtk_One = odeint(valuesAndGradX.dxdt, xTransform(xtNow[i][k]), tNow)[1]
            xtNow[i][k + 1] = (np.array(dxdtk_One)).reshape(2, 1)
            for j in range(ki):
                epstk = y(xtNow[i][k + 1], params) - np.dot(H, xtNow[i][k + 1])
                delta += np.dot(np.dot(epstk.transpose(), pow(R, -1)), epstk)
        Xi += delta
    return Xi[0][0] / 2.0

def y(xtk, params):
    H = params['H']
    R = params['R']
    yEnd = np.dot(H, xtk) + np.random.normal(0, 1) * R
    return yEnd


def main():
    # Определение переменных
    r = 2  # Количество начальных сигналов, альфа
    n = 2  # Размерность вектора х0
    s = 2  # Количество производных по тетта
    N = 30  # Число испытаний
    ki = 1
    q = 1
    m = v = 1.
    tetta = [-1, 0.2]
    valuesAndGradX = gr.gradientX(n, N, tetta)
    F, Psi, H, R, x0, u = valuesAndGradX.initVariables(mode=0)

    params = {"N": N, "ki": ki, "q": q, "F":F, "Psi":Psi, "H":H, "R":R, "x0":x0, "u":u, "valuesAndGradX":valuesAndGradX, "m":m, "v":v}
    print(f'Result is: \n{minimizeFirst(tetta, params)}')



    return 0

main()


