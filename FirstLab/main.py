from scipy.integrate import odeint
import numpy as np
import gradientX as gr


def main():
    # Определение переменных
    r = 2  # Количество начальных сигналов, альфа
    n = 2  # Размерность вектора х0
    s = 2  # Количество производных по тетта
    N = 30  # Число испытаний
    ki = 1
    q = 1
    tetta = (np.array([-2, 0.01])).reshape(2, 1)
    valuesAndGradX = gr.gradientX(n, N, tetta)
    F, Psi, H, R, x0, u = valuesAndGradX.initVariables(mode=0)
    tk = np.arange(N)


    # Поиск производной dxdt:
    x0 = [0, 0]
    tNow = [tk[0], tk[1]]
    dxdtk_One = odeint(valuesAndGradX.dxdt, x0, tNow)[1]
    return 0

main()


