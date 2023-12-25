import numpy as np
import IIMF.CiFile as ci
import IIMF.Variables as variables
import IIntegrate.Integrates as integ
import IDPlan.StartPlan as IStart
import IDPlan.DPlan as IDplan
import IDPlan.Test1ForthPoint as ITest4

def main():
    # Определение переменных
    m = v = nu = 1
    q = 4
    r = 1 # Размерность вектора управления
    n = 2 # Размерность вектора х0
    s = 2 # Количество производных по тетта
    N = 2 # Число испытаний
    tetta_true = np.array([-1.5, 1.0])
    tetta_false = np.array([-1., 1.])
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

    cObject = ci.Ci()

    paramVar = {"F": F, "dF": dF, "Psi": Psi, "dPsi": dPsi, "H": H, "dH": dH, "R": R,
                "dR": dR, "x0": x0, "dx0": dx0, "N": N, "n": n, "t": t, "s": s, "r": r, "q": q}
    paramObj = {"cObject": cObject, "xAObject": xAObject, "dxAObject": dxAObject, "FaObject": FaObject,
                "AtkObject": AtkObject, "dAtkObject": dAtkObject, "eMatrix": eMatrix, "iMatrix": iMatrix}


    #####___Start___#####
    sigm1 = 0.01
    sigm2 = 0.01
    eta = s

    Ksik, matrix = startObj.MainCountPlan(paramVar, paramObj)  # Думаю, matrix можно убрать
    while 1:

        KsikNew = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="dUXMKsik")
        pNew = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="dPXMKsik")
        res = TestForthObj.MainCounter(Ksik["U"], KsikNew, pNew, Ksik["p"], paramVar)
        if res < sigm1:
            print(f"res\t{res}")
            break
        else:
            Ksik["U"] = KsikNew
            Ksik["p"] = pNew
    Ksik["U"] = KsikNew
    Ksik["p"] = pNew
    while 1:
        paramVar["Uk"] = np.random.uniform(0.1, 10., N)
        Uk = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="mu")


        paramVar["Uk"] = Uk
        muNew = DPlanObj.MainDPlan(Ksik, paramVar, paramObj, mode="testMu")
        if (abs(muNew - eta)) <= sigm2:
            print(muNew - eta)
            break
        elif(muNew > eta):
            print("Go to 7 step")
        else:
            continue


    # IMF(params, cObject, xAObject, FaObject, AtkObject, eMatrix, iMatrix)
    # dIMF(params, cObject, xAObject, dxAObject, FaObject, AtkObject, dAtkObject, eMatrix, iMatrix)
#     eMatrix = paramObj["eMatrix"]
#         iMatrix = paramObj["iMatrix"]
#         xAObject = paramObj["xAObject"]
#         dxAObject = paramObj["dxAObject"]
#         FaObject = paramObj["FaObject"]
#         AtkObject = paramObj["AtkObject"]
#         dAtkObject = paramObj["dAtkObject"]
#         cObject = paramObj["cObject"]
#         imfObject = IIMF.IMF()



main()
