import numpy as np

class InitialForFirst:

    def __Variables(self, tetta, params):
        N = params["N"]
        F = np.array([[-0.8, 1.0], [tetta[0], 0]])
        Psi = np.array([[tetta[1]], [1.0]])
        H = np.array([[1.0, 0]])
        R = 0.1
        x0 = np.zeros((2, 1))
        return F, Psi, H, R, x0


    def __dVariables(self, tetta, params):
        N = params["N"]
        s = params["s"]
        dF = np.array([np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 0]])])
        dPsi = np.array([np.array([[0], [0]]), np.array([[1], [0]])])
        dH = np.array([np.array([[0, 0]]), np.array([[0, 0]])])
        dR = np.array([np.array([[0]]), np.array([[0]])])
        dx0 = np.array([np.zeros((2, 1)) for i in range(s)])
        return dF, dPsi, dH, dR, dx0


    def MainVariables(self, tetta, mode, params):
        if mode == "Main":
            F, Psi, H, R, x0 = self.__Variables(tetta, params)
            dF, dPsi, dH, dR, dx0 = self.__dVariables(tetta, params)
            return F, Psi, H, R, x0, dF, dPsi, dH, dR, dx0
