import scipy.io
import numpy as np

# Cargar el archivo .mat
mat = scipy.io.loadmat('/Users/israel_macias/Desktop/xd/threshold/A_star.mat')

# Obtener las matrices a, b, c, y d
a = mat['pathAstar_a'].astype(float)
b = mat['pathAstar_b'].astype(float)
c = mat['pathAstar_a_2'].astype(float)
d = mat['pathAstar_a_2'].astype(float)

# Realizar las operaciones de multiplicación en las columnas correspondientes
a[:, 0] *= 0.05
a[:, 1] *= 0.05

b[:, 0] *= 0.05
b[:, 1] *= 0.05

c[:, 0] *= 0.05
c[:, 1] *= 0.05

d[:, 0] *= 0.05
d[:, 1] *= 0.05

# Formatear los datos y escribir en un archivo .txt
with open('datos.txt', 'w') as file:
    file.write("a_columna1 = [" + ", ".join([f"{val:.2f}" for val in a[:, 0]]) + "]\n")
    file.write("a_columna2 = [" + ", ".join([f"{val:.2f}" for val in a[:, 1]]) + "]\n\n")

    file.write("b_columna1 = [" + ", ".join([f"{val:.2f}" for val in b[:, 0]]) + "]\n")
    file.write("b_columna2 = [" + ", ".join([f"{val:.2f}" for val in b[:, 1]]) + "]\n\n")

    file.write("c_columna1 = [" + ", ".join([f"{val:.2f}" for val in c[:, 0]]) + "]\n")
    file.write("c_columna2 = [" + ", ".join([f"{val:.2f}" for val in c[:, 1]]) + "]\n\n")

    file.write("d_columna1 = [" + ", ".join([f"{val:.2f}" for val in d[:, 0]]) + "]\n")
    file.write("d_columna2 = [" + ", ".join([f"{val:.2f}" for val in d[:, 1]]) + "]\n")
