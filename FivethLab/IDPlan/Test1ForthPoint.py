import numpy as np

class Test1ForthPoint:
    def MainCounter(self, Uk, UkPlus1, pk, pPlus1, paramVar):
        q = paramVar["q"]
        N = paramVar["N"]
        UkLine = self.__LineU(Uk)
        UkPlus1Line = self.__LineU(UkPlus1)
        result = 0.

        l2res = self.__L2Norm(UkPlus1Line, UkLine)
        for i in range(q):
            result += (l2res + pow(pPlus1[i] - pk[i], 2))
        return result


    def __L2Norm(self, a1, a2):
        result = 0.
        for i in range(len(a1)):
            result += pow(a1[i] - a2[i], 2)
        return result


    def __LineU(self, U):
        result = []
        for line in U:
            for element in line:
                result.append(element[0][0])
        return result

