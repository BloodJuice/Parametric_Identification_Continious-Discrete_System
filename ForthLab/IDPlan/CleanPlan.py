import numpy as np

class CleanPlan:
    def LineU(self, U):
        result = []
        for line in U:
            for element in line:
                result.append(element[0][0])
        return result

    def RechargeUkPk(self, tk, U, Uk, pk):
        ksikNew = np.hstack((U, Uk))
        n = len(pk)
        for i in range(n):
            pk[i] = abs(1. - tk) * pk[i]
        pNew = np.hstack((pk, tk))
        return ksikNew, pNew

    def CleaningPlan(self, Ksik, p, delta, N):
        # Функция получает Ksik в виде вектора*
        newp = []  # p, содержащая только веса, которые встречаются больше двух раз
        newKsik = []
        pDelPoint = []
        Q = len(p)

        Ksik = Ksik.reshape(Q, N, 1)
        for stepi in range(Q - 1):
            for stepj in range(stepi + 1, Q - stepi):
                if np.dot(Ksik[stepi].transpose(), Ksik[stepj]) <= delta:
                    p[stepi] += p[stepj]
                    pDelPoint.append(stepj)

        for stepi in range(Q):
            if p[stepi] < delta:
                pDelPoint.append(stepi)

        for stepi in range(Q):
            if stepi not in pDelPoint:
                newKsik.append(Ksik[stepi])
                newp.append(p[stepi])

        # Уравновешиваю веса, приводя их сумму к 1:
        pSum = sum(newp)
        if pSum != (1.0 - delta) or pSum != (1.0 + delta):
            for stepi in range(len(newp)):
                newp[stepi] *= (1.0 / pSum)

        newKsik = (np.array(newKsik)).reshape(len(newp), N, 1)
        newp = np.array(newp)
        return newKsik, newp

