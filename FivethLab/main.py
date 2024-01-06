import math

import numpy as np
import IIMF.CiFile as ci
import IIMF.Variables as variables
import IIntegrate.Integrates as integ
import IDPlan.StartPlan as IStart
import IDPlan.DPlan as IDplan
import IDPlan.Test1ForthPoint as ITest4
import IDPlan.CleanPlan as ICleanPlan
import first as First




def main():
    # Определение переменных
    m = nu = 1
    q = 4
    r = 1 # Размерность вектора управления
    n = 2 # Размерность вектора х0
    s = 2 # Количество производных по тетта
    N = 4 # Число испытаний
    ki = [1. for i in range(q)]
    v = sum(ki)
    uStart = np.array([[1.] for i in range(q)])

    tetta_true = np.array([-1.5, 1.0])
    tetta_false = np.array([-0.5, 0.05])
    t = []

    if n == 1:
        count = 0
        param = [1, 1, 1, 2] # Reshape params for test
        for i in range(N + 1):
            if i == 0:
                t.append(count)
            else:
                count += 0.5
                t.append(count)
    else:
        param = [2, 2, 2, 1] # F (2, 2) and Psi (2, 1)
        t = np.arange(N + 1)

    varObject = variables.IMFVariablesSaver()

    # Choose Test or NoTest
    F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u = varObject.Variables(tetta_true, N, "IMF", modeTest=n)

    eMatrix = integ.TransisionMatrix()
    eMatrix.rP = param
    iMatrix = integ.IntegrateMethods()
    iMatrix.u = u
    xAObject = integ.Xatk(F, dF, Psi, dPsi, t=[t[0], t[1]], x0=x0, dx0=dx0)
    dxAObject = integ.dXatk(F, dF, Psi, dPsi, t=[t[0], t[1]], x0=x0, dx0=dx0)
    FaObject = integ.Fatk(F, dF)
    AtkObject = integ.Atk(F, dF, Psi, dPsi, x0=x0, dx0=dx0)
    dAtkObject = integ.dAtk(F, dF, Psi, dPsi, x0=x0, dx0=dx0, N=N)
    startObj = IStart.StartPlan()
    DPlanObj = IDplan.DPlan()
    TestForthObj = ITest4.Test1ForthPoint()
    CleanPlanObj = ICleanPlan.CleanPlan()

    cObject = ci.Ci()

    paramVar = {"F": F, "dF": dF, "Psi": Psi, "dPsi": dPsi, "H": H, "dH": dH, "R": R,
                "dR": dR, "x0": x0, "dx0": dx0, "N": N, "n": n, "t": t, "s": s, "r": r, "q": q}
    paramObj = {"cObject": cObject, "xAObject": xAObject, "dxAObject": dxAObject, "FaObject": FaObject,
                "AtkObject": AtkObject, "dAtkObject": dAtkObject, "eMatrix": eMatrix, "iMatrix": iMatrix}

    # START отсюда
    Ksik, matrix = startObj.MainCountPlan(paramVar, paramObj)  # Думаю, matrix можно убрать

    params = {"m": m, "v": v, "nu": 1, "q": q, "r": r, "n": n, "s": s, "N": N, "ki": ki, "u": Ksik["U"],
              "tetta_true": tetta_true, "tetta_false": tetta_false}
    tettaNew = First.first(params)
    params["tetta_false"] = tettaNew
    KsikNew, pNew = ForthLab(params)
    print(f"\ntettaOld:\t{tettaNew}")


    ki, v = rounding(pNew, params)

    params["ki"] = ki
    params["v"] = v
    params["u"] = KsikNew
    params["q"] = len(pNew)
    tettaNew = First.first(params)
    print(f"\ntettaNew:\t{tettaNew}")
    return 0

def ForthLab(params):

    m = params["m"]
    v = params["v"]
    nu = params["nu"]
    q = params["q"]
    r = params["r"] # Размерность вектора управления
    n = params["n"] # Размерность вектора х0
    s = params["s"] # Количество производных по тетта
    N = params["N"] # Число испытаний

    tetta_true = params["tetta_true"]
    tetta_false = params["tetta_false"]
    t = []

    if n == 1:
        count = 0
        param = [1, 1, 1, 2] # Reshape params for test
        for i in range(N + 1):
            if i == 0:
                t.append(count)
            else:
                count += 0.5
                t.append(count)
    else:
        param = [2, 2, 2, 1] # F (2, 2) and Psi (2, 1)
        t = np.arange(N + 1)

    varObject = variables.IMFVariablesSaver()

    # Choose Test or NoTest
    F, dF, Psi, dPsi, H, dH, R, dR, x0, dx0, u = varObject.Variables(tetta_false, N, "IMF", modeTest=n)

    eMatrix = integ.TransisionMatrix()
    eMatrix.rP = param
    iMatrix = integ.IntegrateMethods()
    iMatrix.u = u
    xAObject = integ.Xatk(F, dF, Psi, dPsi, t=[t[0], t[1]], x0=x0, dx0=dx0)
    dxAObject = integ.dXatk(F, dF, Psi, dPsi, t=[t[0], t[1]], x0=x0, dx0=dx0)
    FaObject = integ.Fatk(F, dF)
    AtkObject = integ.Atk(F, dF, Psi, dPsi, x0=x0, dx0=dx0)
    dAtkObject = integ.dAtk(F, dF, Psi, dPsi, x0=x0, dx0=dx0, N=N)
    startObj = IStart.StartPlan()
    DPlanObj = IDplan.DPlan()
    TestForthObj = ITest4.Test1ForthPoint()
    CleanPlanObj = ICleanPlan.CleanPlan()

    cObject = ci.Ci()

    paramVar = {"F": F, "dF": dF, "Psi": Psi, "dPsi": dPsi, "H": H, "dH": dH, "R": R,
                "dR": dR, "x0": x0, "dx0": dx0, "N": N, "n": n, "t": t, "s": s, "r": r, "q": q}
    paramObj = {"cObject": cObject, "xAObject": xAObject, "dxAObject": dxAObject, "FaObject": FaObject,
                "AtkObject": AtkObject, "dAtkObject": dAtkObject, "eMatrix": eMatrix, "iMatrix": iMatrix}


    #####___Start___#####
    sigm1 = 0.001
    sigm2 = 0.001
    eta = s

    Ksik, matrix = startObj.MainCountPlan(paramVar, paramObj)  # Думаю, matrix можно убрать
    print(Ksik["U"])
    XMStart = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="countTestPlan")
    while 1:

        KsikNew = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="dUXMKsik")
        pNew = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="dPXMKsik")
        res = TestForthObj.MainCounter(Ksik["U"], KsikNew, pNew, Ksik["p"], paramVar)
        if res < sigm1:
            print(f"res\t{res}")
            print(f"Let's go to next page")
            break
        else:
            print(f"res\t{res}")
            Ksik["U"] = KsikNew
            Ksik["p"] = pNew
    Ksik["U"] = KsikNew
    Ksik["p"] = pNew
    while 1:
        paramVar["Uk"] = np.random.uniform(0.1, 10., N)
        Uk = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="mu")

        paramVar["Uk"] = Uk
        muNew = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="testMu")
        if (abs(muNew - eta)) < sigm2 or math.isclose(abs(muNew - eta), sigm2):
            paramVar["Uk"] = Uk
            tk = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="tkSearcher")
            KsikLine = CleanPlanObj.LineU(Ksik["U"])
            KsikNew, pNew = CleanPlanObj.RechargeUkPk(tk, KsikLine, Uk, pNew)
            KsikNew, pNew = CleanPlanObj.CleaningPlan(KsikNew, pNew, 0.01, N)
            Ksik["U"] = KsikNew.copy()
            Ksik["p"] = pNew
            paramVar["q"] = len(pNew)
            XMEnd = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="countTestPlan")
            break
        elif(muNew > eta):
            print(muNew - eta)
            paramVar["Uk"] = Uk
            tk = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="tkSearcher")
            KsikLine = CleanPlanObj.LineU(Ksik["U"])
            KsikNew, pNew = CleanPlanObj.RechargeUkPk(tk, KsikLine, Uk, pNew)
            KsikNew, pNew = CleanPlanObj.CleaningPlan(KsikNew, pNew, 0.01, N)
            Ksik["U"] = KsikNew.copy()
            Ksik["p"] = pNew
            paramVar["q"] = len(pNew)

            print(f"KsikNew: {KsikNew}"
                  f"\npNew: {pNew}\n")
            continue
        else:
            continue
    print(f"__________________THE_END!!!______________________\n"
          f"KsikNew: {KsikNew}"
          f"\npNew: {pNew}")
    print(f"XMStart:\t{XMStart}\n"
          f"XMEnd:\t{XMEnd}")
    return KsikNew, pNew



def rounding(pNew, params):
    sigmHatch, sigmTwiceHatch, vHatch, vTwiceHatch, sigm, v1, pointThree = [], [], 0, 0, [], 0, []
    q, v = len(pNew), 10
    for i in range(q):
        sigmHatch.append((v - q) * pNew[i])
        sigmTwiceHatch.append(int(v * pNew[i]))
    # print("\nsigmHatch:\n", sigmHatch, "\nsigmTwiceHatch:\n", sigmTwiceHatch)
    vHatch, vTwiceHatch  = v, v
    for i in range(q):
        vHatch -= sigmHatch[i]
        vTwiceHatch -= sigmTwiceHatch[i]

    # Point 2
    if vHatch < vTwiceHatch:
        for i in range(int(q)):
            sigm.append(sigmHatch[i])
        v1 = vHatch
    else:
        for i in range(int(q)):
            sigm.append(sigmTwiceHatch[i])
        v1 = vTwiceHatch

    # Point 3
    for i in range(int(q)):
        pointThree.append(v * pNew[i] - sigm[i])
    pointThree = sorted(pointThree, reverse=True)
    # print("\npointThree:\n", pointThree, "\nsigm\n", sigm, "\nv1:\n", v1)

    s = np.zeros(q)
    for i in range(int(v1)):
        for j in range(int(q)):
            if pointThree[i] == (v * pNew[j] - sigm[j]):
                s[j] = 1
            else:
                s[j] = 0
    kNew = np.zeros(int(q))
    for i in range(int(q)):
        kNew[i] = sigm[i] + s[i]
    return kNew, v


main()
