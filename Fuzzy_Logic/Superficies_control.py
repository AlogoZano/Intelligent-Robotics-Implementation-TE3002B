import numpy as np
import skfuzzy as sk
from matplotlib import pyplot as plt


dist = np.arange(0.0, 10.1, 0.1)
orient = np.arange(-np.pi/2, (np.pi/2)+0.01, 0.01)
vel_lin = np.arange(0.0, 5.1, 0.1)
vel_ang = np.arange(-5.0, 5.1, 0.1)

sigma_1 = 1.0
sigma_2 = 1.0
sigma_3 = 1.0

dist_short = sk.gaussmf(dist, 0.0, sigma_1)
dist_short_med = sk.gaussmf(dist, 2.5, sigma_3)
dist_med = sk.gaussmf(dist, 5.0, sigma_2)
dist_long_med = sk.gaussmf(dist, 7.5, sigma_3)
dist_long = sk.gaussmf(dist, 10.0, sigma_1)

sigma_1 = 0.32
sigma_2 = 0.32
sigma_3 = 0.32

or_left = sk.gaussmf(orient, -np.pi/2, sigma_1)
or_left_med = sk.gaussmf(orient, -np.pi/4, sigma_3)
or_min = sk.gaussmf(orient, 0.0, sigma_2)
or_right_med = sk.gaussmf(orient, np.pi/4, sigma_3)
or_right = sk.gaussmf(orient, np.pi/2, sigma_1)


vel_lin_low = sk.trimf(vel_lin, [0.0, 0.0, 1.25])
vel_lin_med_low = sk.trimf(vel_lin, [0.75, 1.75, 2.75])
vel_lin_med = sk.trimf(vel_lin, [1.75, 2.75, 3.75])
vel_lin_med_high = sk.trimf(vel_lin, [2.75, 3.75, 4.75])
vel_lin_high = sk.trimf(vel_lin, [3.75, 5.0, 5.0])

vel_high_cw = sk.trimf(vel_ang, [-5.0, -5.0, -2.5])
vel_low_cw = sk.trimf(vel_ang, [-5.0, -2.5, 0.0])
vel_low = sk.trimf(vel_ang, [-2.5, 0.0, 2.5])
vel_low_ccw = sk.trimf(vel_ang, [0.0, 2.5, 5.0])
vel_high_ccw = sk.trimf(vel_ang, [2.5, 5.0, 5.0])


def fuzzy_inference_vel_lin(dist_value, orient_value):
   
    mu_dist_short = sk.interp_membership(dist, dist_short, dist_value)
    mu_dist_short_med = sk.interp_membership(dist, dist_short_med, dist_value)
    mu_dist_med = sk.interp_membership(dist, dist_med, dist_value)
    mu_dist_long_med = sk.interp_membership(dist, dist_long_med, dist_value)
    mu_dist_long = sk.interp_membership(dist, dist_long, dist_value)

    mu_or_left = sk.interp_membership(orient, or_left, orient_value)
    mu_or_left_med = sk.interp_membership(orient, or_left_med, orient_value)
    mu_or_min = sk.interp_membership(orient, or_min, orient_value)
    mu_or_right_med = sk.interp_membership(orient, or_right_med, orient_value)
    mu_or_right = sk.interp_membership(orient, or_right, orient_value)

    #viene lo bueno :0
    R1 = min(mu_dist_long, mu_or_left) #media baja
    R2 = min(mu_dist_long, mu_or_left_med) #media baja
    R3 = min(mu_dist_long, mu_or_min) #baja
    R4 = min(mu_dist_long, mu_or_right_med) #media baja
    R5 = min(mu_dist_long, mu_or_right) #media baja
    
    R6 = min(mu_dist_long_med, mu_or_left) #media baja
    R7 = min(mu_dist_long_med, mu_or_left_med) #media baja
    R8 = min(mu_dist_long_med, mu_or_min) #media baja
    R9 = min(mu_dist_long_med, mu_or_right_med) #media baja
    R10 = min(mu_dist_long_med, mu_or_right) #media baja
    
    R11 = min(mu_dist_med, mu_or_left) #media alta
    R12 = min(mu_dist_med, mu_or_left_med) #media 
    R13 = min(mu_dist_med, mu_or_min) #media
    R14 = min(mu_dist_med, mu_or_right_med) #media
    R15 = min(mu_dist_med, mu_or_right) #media alta
    
    R16 = min(mu_dist_short_med, mu_or_left) #media alta
    R17 = min(mu_dist_short_med, mu_or_left_med) #media alta
    R18 = min(mu_dist_short_med, mu_or_min) #media alta
    R19 = min(mu_dist_short_med, mu_or_right_med) #media alta
    R20 = min(mu_dist_short_med, mu_or_right) #media alta
    
    R21 = min(mu_dist_short, mu_or_left) #alta
    R22 = min(mu_dist_short, mu_or_left_med) #alta
    R23 = min(mu_dist_short, mu_or_min) #alta
    R24 = min(mu_dist_short, mu_or_right_med) #alta
    R25 = min(mu_dist_short, mu_or_right) #alta

    vel_lin_activations = {
        'baja': max(R16,R20,R21,R23,R25),
        'media_baja': max(R17,R18,R19,R22,R24),
        'media': max(R6,R10,R11,R12,R13,R14,R15),
        'media_alta': max(R1,R2,R4,R5,R7,R9),
        'alta': max(R3,R8)
    }

   
    aggregated = np.zeros_like(vel_lin)
    aggregated = np.fmax(aggregated, np.fmin(vel_lin_low, vel_lin_activations['baja']))
    aggregated = np.fmax(aggregated, np.fmin(vel_lin_med_low, vel_lin_activations['media_baja']))
    aggregated = np.fmax(aggregated, np.fmin(vel_lin_med, vel_lin_activations['media']))
    aggregated = np.fmax(aggregated, np.fmin(vel_lin_med_high, vel_lin_activations['media_alta']))
    aggregated = np.fmax(aggregated, np.fmin(vel_lin_high, vel_lin_activations['alta']))

    vel_lin_defuzz = sk.defuzz(vel_lin, aggregated, 'centroid')
    
    return vel_lin_defuzz, aggregated

def fuzzy_inference_vel_ang(dist_value, orient_value):
   
    mu_dist_short = sk.interp_membership(dist, dist_short, dist_value)
    mu_dist_short_med = sk.interp_membership(dist, dist_short_med, dist_value)
    mu_dist_med = sk.interp_membership(dist, dist_med, dist_value)
    mu_dist_long_med = sk.interp_membership(dist, dist_long_med, dist_value)
    mu_dist_long = sk.interp_membership(dist, dist_long, dist_value)

    mu_or_left = sk.interp_membership(orient, or_left, orient_value)
    mu_or_left_med = sk.interp_membership(orient, or_left_med, orient_value)
    mu_or_min = sk.interp_membership(orient, or_min, orient_value)
    mu_or_right_med = sk.interp_membership(orient, or_right_med, orient_value)
    mu_or_right = sk.interp_membership(orient, or_right, orient_value)

    #viene lo bueno pt. 2 :0
    R1 = min(mu_dist_short, mu_or_left) #izq
    R2 = min(mu_dist_short, mu_or_left_med) #izq
    R3 = min(mu_dist_short, mu_or_min) #min
    R4 = min(mu_dist_short, mu_or_right_med) #der
    R5 = min(mu_dist_short, mu_or_right) #der
    
    R6 = min(mu_dist_short_med, mu_or_left) #izq
    R7 = min(mu_dist_short_med, mu_or_left_med) #izq
    R8 = min(mu_dist_short_med, mu_or_min) #min
    R9 = min(mu_dist_short_med, mu_or_right_med) #der
    R10 = min(mu_dist_short_med, mu_or_right) #der
    
    R11 = min(mu_dist_med, mu_or_left) #med izq
    R12 = min(mu_dist_med, mu_or_left_med) #med izq
    R13 = min(mu_dist_med, mu_or_min) #min
    R14 = min(mu_dist_med, mu_or_right_med) #med der
    R15 = min(mu_dist_med, mu_or_right) #med der
    
    R16 = min(mu_dist_long_med, mu_or_left) #med izq
    R17 = min(mu_dist_long_med, mu_or_left_med) #izq
    R18 = min(mu_dist_long_med, mu_or_min) #min
    R19 = min(mu_dist_long_med, mu_or_right_med) #der
    R20 = min(mu_dist_long_med, mu_or_right) #med der
    
    R21 = min(mu_dist_long, mu_or_left) #med izq
    R22 = min(mu_dist_long, mu_or_left_med) #med izq
    R23 = min(mu_dist_long, mu_or_min) #min
    R24 = min(mu_dist_long, mu_or_right_med) #med der
    R25 = min(mu_dist_long, mu_or_right) #med der

    vel_ang_activations = {
        'izq': max(R1, R6, R11, R16,R21),
        'med_izq': max(R2, R7, R12, R17,R22),
        'min': max(R3, R8, R13, R18, R23),
        'med_der': max(R4, R9, R14, R19, R24),
        'der': max(R5, R10, R15, R20,R25)
    }

   
    aggregated = np.zeros_like(vel_ang)
    aggregated = np.fmax(aggregated, np.fmin(vel_high_ccw, vel_ang_activations['izq']))
    aggregated = np.fmax(aggregated, np.fmin(vel_low_ccw, vel_ang_activations['med_izq']))
    aggregated = np.fmax(aggregated, np.fmin(vel_low, vel_ang_activations['min']))
    aggregated = np.fmax(aggregated, np.fmin(vel_low_cw, vel_ang_activations['med_der']))
    aggregated = np.fmax(aggregated, np.fmin(vel_high_cw, vel_ang_activations['der']))

    vel_ang_defuzz = sk.defuzz(vel_ang, aggregated, 'centroid')
    
    return vel_ang_defuzz, aggregated

dist_value = 9.821  
orient_value = -0.8

# PRUEBA DE VELOCIDAD LINEAL
velocidad_lineal, aggregated_lin = fuzzy_inference_vel_lin(dist_value, orient_value)
print("Velocidad lineal inferida:", velocidad_lineal)

#PRUEBA DE VELOCIDAD ANGULAR
velocidad_angular, aggregated_ang = fuzzy_inference_vel_ang(dist_value, orient_value)
print("Velocidad angular inferida:", velocidad_angular)

##################################### SUPERFICIES DE CONTROL Y CENTROIDE #####################################

#SUPERFICIE DE CONTROL DE VELOCIDAD LINEAL
plt.figure(1)
plt.plot(vel_lin, vel_lin_low, 'b', linewidth=1.5, label='Baja')
plt.plot(vel_lin, vel_lin_med_low, 'g', linewidth=1.5, label='Media baja')
plt.plot(vel_lin, vel_lin_med, 'r', linewidth=1.5, label='Media')
plt.plot(vel_lin, vel_lin_med_high, 'c', linewidth=1.5, label='Media alta')
plt.plot(vel_lin, vel_lin_high, 'm', linewidth=1.5, label='Alta')
plt.fill_between(vel_lin, np.zeros_like(vel_lin), aggregated_lin, facecolor='Purple', alpha=0.4)
plt.plot([velocidad_lineal, velocidad_lineal], [0, sk.interp_membership(vel_lin, aggregated_lin, velocidad_lineal)], 'k', linewidth=1.5, alpha=0.9)
plt.title('Agregado de salidas y resultado defuzzificado (centroide)')
plt.xlabel('Velocidad lineal')
plt.ylabel('Grado de membresía')
plt.legend()

plt.show()

s = (len(orient), len(dist))  
cs_lin = np.zeros(s)
cs_ang = np.zeros(s)

for j in range(len(dist)):
    for i in range(len(orient)):
        cs_lin[i, j], _ = fuzzy_inference_vel_lin(dist[j], orient[i])
        cs_ang[i, j], _ = fuzzy_inference_vel_ang(dist[j], orient[i]) 
X, Y = np.meshgrid(dist, orient)
Z_lin = np.array(cs_lin)
Z_ang = np.array(cs_ang)


plt.figure(2)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z_lin,
                rstride=1, 
                cstride=1, 
                cmap='inferno', 
                edgecolor='none')

ax.set_xlabel("Distancia")
ax.set_ylabel("Orientación")
ax.set_zlabel("Velocidad lineal")
plt.title("Superficie de control (velocidad lineal)")



#SUPERFICIE DE CONTROL DE VELOCIDAD ANGULAR
y_value = sk.interp_membership(vel_ang, aggregated_ang, velocidad_angular)

# SUPERFICIE DE CONTROL DE VELOCIDAD ANGULAR
plt.figure(3)
plt.plot(vel_ang, vel_high_ccw, 'b', linewidth=1.5, label='izq')
plt.plot(vel_ang, vel_low_ccw, 'g', linewidth=1.5, label='med izq')
plt.plot(vel_ang, vel_low, 'r', linewidth=1.5, label='min')
plt.plot(vel_ang, vel_low_cw, 'c', linewidth=1.5, label='med der')
plt.plot(vel_ang, vel_high_cw, 'm', linewidth=1.5, label='der')
plt.fill_between(vel_ang, np.zeros_like(vel_ang), aggregated_ang, facecolor='Purple', alpha=0.3)

# Plotting the cross
plt.scatter(velocidad_angular, y_value, color='k', marker='x', s=100, linewidths=1.5, alpha=0.9)

plt.title('Agregado de salidas y resultado defuzzificado (centroide)')
plt.xlabel('Velocidad angular')
plt.ylabel('Grado de membresía')
plt.legend()

plt.show()


plt.figure(4)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z_ang,
                rstride=1, 
                cstride=1, 
                cmap='inferno', 
                edgecolor='none')

ax.set_xlabel("Distancia")
ax.set_ylabel("Orientación")
ax.set_zlabel("Velocidad angular")
plt.title("Superficie de control (velocidad angular)")
plt.show()