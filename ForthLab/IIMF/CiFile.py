import numpy as np

class Ci:
    def CountCi(self, I, n, s):
        self.matrix_ci = np.zeros((n, n * (s + 1)))
        for _ in range(n):
            self.matrix_ci[_][I * n + _] = 1
        return self.matrix_ci
    def ReturnCi(self):
        return self.matrix_ci