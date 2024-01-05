import numpy as np

# return F, Psi, H, R, x0, u
class IMFVariablesSaver:
    # modes: IMF, IMF_test, dIMF, dIMF_test
    def Variables(self, tetta, N, modeName, modeTest):
        # return: F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u
        if modeName == "IMF" and modeTest == 2:
            self.F = np.array([-0.8, 1., tetta[0], 0.])
            self.dF = [np.array([0., 0., 1., 0.]), np.array([0., 0., 0., 0.])]
            self.Psi = np.array([tetta[1], 1.])
            self.dPsi = [np.array([0., 0.]), np.array([1., 0.])]
            self.H = np.array([1., 0.])
            self.dH = [np.array([0., 0.]), np.array([0., 0.])]
            self.R = 0.1
            self.dR = [0., 0.]
            self.x0 = np.array([[0.], [0.]])
            self.dx0 = [np.array([[0.], [0.]]), np.array([[0.], [0.]])]
            self.u = np.array([1. for i in range(N)])
            return self.F, self.dF, self.Psi, self.dPsi, self.H, self.dH, self.R, self.dR, self.x0, self.dx0, self.u
        elif modeName == "IMF" and modeTest == 1:
            self.F = np.array([[0.]])
            self.dF = [np.array([[0.]]), np.array([[0.]])]
            self.Psi = np.array([[tetta[0], tetta[1]]])
            self.dPsi = [np.array([[1., 0.]]), np.array([[0., 1.]])]
            self.H = np.array([1.])
            self.dH = [np.array([0.]), np.array([0.])]
            self.R = 0.3
            self.dR = [0., 0.]
            self.x0 = np.zeros((1, 1))
            self.dx0 = [np.zeros((1, 1)) for i in range(2)]
            self.u = np.array([[[1.], [1.]] for i in range(N + 1)])
            return self.F, self.dF, self.Psi, self.dPsi, self.H, self.dH, self.R, self.dR, self.x0, self.dx0, self.u
