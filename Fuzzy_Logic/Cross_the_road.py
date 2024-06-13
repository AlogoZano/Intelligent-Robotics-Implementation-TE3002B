import numpy as np
import skfuzzy as sk
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


vel_a = np.arange(0, 121, 1)
dist_a = np.arange(20, 101, 1)
vel_p = np.arange(1, 13.1, 0.1)

def fis(va, da):

    #Conjuntos de variable velocidad del auto
    vel_a_lo = sk.trimf(vel_a, [0, 0, 45])
    vel_a_me = sk.trimf(vel_a, [30, 60, 90])
    vel_a_hi = sk.trimf(vel_a, [75, 120, 120])

    # plt.figure(1)
    # plt.subplot(131)
    # plt.plot(vel_a, vel_a_lo, label='baja')
    # plt.plot(vel_a, vel_a_me, label='media')
    # plt.plot(vel_a, vel_a_hi, label='alta')
    # plt.title("Velocidades del auto")
    # plt.xlabel("km/h")
    # plt.ylabel(r'$\mu$')
    # plt.legend()

    #Conjuntos de variable distancia
    dist_a_short = sk.trapmf(dist_a, [20, 20, 35, 45])
    dist_a_mod = sk.trapmf(dist_a, [40, 50, 70, 80])
    dist_a_long = sk.trapmf(dist_a, [75, 85, 100, 100])


    # plt.subplot(132)
    # plt.plot(dist_a, dist_a_short, label='corta')
    # plt.plot(dist_a, dist_a_mod, label='moderada')
    # plt.plot(dist_a, dist_a_long, label='larga')
    # plt.title("Distancia al auto")
    # plt.xlabel("m")
    # plt.ylabel(r'$\mu$')
    # plt.legend()


    #Conjuntos de variable velocidad del peatón
    vel_p_low = sk.gbellmf(vel_p, 1, 3, 1)
    vel_p_mlow = sk.gbellmf(vel_p, 1, 3, 4)
    vel_p_me = sk.gbellmf(vel_p, 1, 3, 7)
    vel_p_mhigh = sk.gbellmf(vel_p, 1, 3, 10)
    vel_p_high = sk.gbellmf(vel_p, 1, 3, 13)

    # plt.subplot(133)
    # plt.plot(vel_p, vel_p_low, label='baja')
    # plt.plot(vel_p, vel_p_mlow, label='media baja')
    # plt.plot(vel_p, vel_p_me, label='media')
    # plt.plot(vel_p, vel_p_mhigh, label='media alta')
    # plt.plot(vel_p, vel_p_high, label='alta')
    # plt.title("Velocidades del peatón")
    # plt.xlabel("km/h")
    # plt.ylabel(r'$\mu$')
    # plt.legend()

    #Entrada
    # va = int(input("Introduzca velocidad del auto: "))
    # da = int(input("Introduzca distancia del auto: "))

    #Reglas de inferencia
    R1 = min(vel_a_lo[va], dist_a_long[da-20])
    R2 = min(vel_a_me[va], dist_a_long[da-20])
    R3 = min(vel_a_hi[va], dist_a_long[da-20])

    R4 = min(vel_a_lo[va], dist_a_mod[da-20])
    R5 = min(vel_a_me[va], dist_a_mod[da-20])
    R6 = min(vel_a_hi[va], dist_a_mod[da-20])

    R7 = min(vel_a_lo[va], dist_a_short[da-20])
    R8 = min(vel_a_me[va], dist_a_short[da-20])
    R9 = min(vel_a_hi[va], dist_a_short[da-20])

    #print(R1, "\n", R2, "\n", R3, "\n", R4, "\n", R5, "\n", R6, "\n", R7, "\n", R8, "\n", R9)

    max_vel_p_low = max(R1, 0)
    max_vel_p_mlow = max(R2, R4)
    max_vel_p_me = max(R3, R5, R7)
    max_vel_p_mhigh = max(R6, R8)
    max_vel_p_high = max(R9, 0)

    #print(max_vel_p_low, "\n", max_vel_p_mlow, "\n", max_vel_p_me, "\n", max_vel_p_mhigh, "\n", max_vel_p_high)

    arc = []

    num = 0
    den = 0

    for i in range (len(vel_p)):
        vel_p_low[i] = min(vel_p_low[i], max_vel_p_low)
        vel_p_mlow[i] = min(vel_p_mlow[i], max_vel_p_mlow)
        vel_p_me[i] = min(vel_p_me[i], max_vel_p_me)
        vel_p_mhigh[i] = min(vel_p_mhigh[i], max_vel_p_mhigh)
        vel_p_high[i] = min(vel_p_high[i], max_vel_p_high)
        arc.append(max(vel_p_low[i], vel_p_mlow[i],vel_p_me[i],vel_p_mhigh[i],vel_p_high[i]))

    for i in range(len(vel_p)):
        num += vel_p[i]*arc[i]
        den += arc[i]
    
    defuzz = round((num/den), 2)
    
    # print("La velocidad del peaton debe de ser aprox", defuzz, "km/h")    
    # plt.figure(2)
    # plt.plot(vel_p,arc,color='black',label='agregado')
    # plt.title("agregado")
    # plt.xlabel("velocidad del peaton")
    # plt.ylabel(r'$\mu$')
    #plt.show()
    
    return (defuzz)
    
s = 121, 81
cs  =np.zeros(s)
for j in range(len(dist_a)):
    for i in range(len(vel_a)):
        cs[i,j] = fis(vel_a[i],dist_a[j])
X,Y = np.meshgrid(dist_a,vel_a)
Z = np.array(cs)


plt.figure(1)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z,
                rstride=1, 
                cstride=1, 
                cmap='inferno', 
                edgecolor='none')

ax.set_xlabel("distancia del auto")
ax.set_ylabel("velocidad del auto")
ax.set_zlabel("velocidad del peaton")
plt.title("Superficie de control")
plt.show()

        

