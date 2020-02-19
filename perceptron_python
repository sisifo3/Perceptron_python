
import pandas as pd
import numpy as np
import random
from random import *
import matplotlib.pyplot as plt


df = pd.read_csv("test.txt", sep = " ") #import test and read and its separete por " "
df.tail()

x = df.iloc[:, 0:1].values	
y = df.iloc[:, 1:2].values
p = df.iloc[:, 0:2].values
e = df.iloc[:, 2:3].values
w = np.random.rand(1,2)

b = 0.5
t = df.iloc[:, 2:3].values
z = np.empty([49,2])
o = np.array([0,0])
error = 0
error2=0
x3 = 6
trainfor50 = 0
for ite in range(8):
	for n in range(50):
		#lamb_da = np.dot(w,p[n])
		lamb_da = (np.dot(w,p[n])) + b#multiplicación de pesos por caracteristicas
		#print(lamb_da)
		
		if(lamb_da > 0): #si es mayor a 0 o menor 
			lamb_da = 1
		else:
			lamb_da = 0
		w = w + ((t[n]-lamb_da)*p[n])#El nuevo peso 
		b = b +(t[n]-lamb_da)#el nuevo bias
		#np.concatenate((z,w))			#guardamos los pesos en una matriz 'z' de 50x2
		#print("corregido : " )
		#print(lamb_da)
		if lamb_da == e[n]:
		#	print("corect")
		#	print(lamb_da,e[n])
			error2 = error2+1#sumamos los errores para poder sacar el porcentaje
		else:
		#	print("incorrect")	
		#	print(lamb_da,e[n])
			error = error +1 #sumamos los errores para poder sacar el porcentaje
	
		#etiqueta_actual = (np.dot(w,p[n]))
		#print(etiqueta_actual)
		
	trainfor50 =50 + (50 * ite)	
#y3 = -((w[:,0]/w[:,1])*(x3))
	

correct_porcentage = ((trainfor50 - error)*100)/trainfor50 #Para obtener el porciento de error
print(correct_porcentage)	
#print(x3,y3)
print(w)
#print(b)

x1 = [0,w[0:1,1]]
y1 = [0,w[0:1,0]]

#wn = np.rot90(w)
#wn = np.rot90(wn)
x2 = (w[0:1,1])*-1
y2 = y1
#print(x1)
#print(x2)
#print(y2)
#print(wn)
#print(wn[:,0])

#plt.plot(range(5))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.draw()

y4 = w[:,0]
x4 = -(w[:,1])
plt.plot(y1,x1,	 linestyle='-')  # solid
plt.plot([0,x2],y2,	 linestyle='-')  # solid
plt.scatter(x4,y4, color='black', marker='x', label='datos')	#Los pesos pero invertidos y con un signo cambiado
plt.scatter(w[:,0],w[:,1], color='black', marker='x', label='datos')  #el punto donde se encuentra los ultimos pesos

for n in range(50):															#Ciclo for para imprimir las dos caracteristicas
	if(e[n] == 1):
		plt.scatter(x[n],y[n], color='red', marker='o', label='datos')		
	else:
		plt.scatter(x[n],y[n], color='blue', marker='o', label='datos')
#plt.scatter(wn[0:1,0],wn[1:2,0], color='blue', marker='#', label='pesos')
#plt.scatter(x,y, color='red', marker='o', label='datos')
plt.scatter(0,0, color='green', marker='*', label='datos')
plt.show()
#print(e)
