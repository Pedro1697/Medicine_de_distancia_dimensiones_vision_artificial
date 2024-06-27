from mpl_toolkits import mplot3d
import numpy as np 
import matplotlib.pyplot as plt
from skimage import data, filters,color,io
import pandas as pd 

file=pd.ExcelFile('Manta.xlsx')
df=file.parse('Hoja2')
dfa=np.array(df)

x=dfa[1::,0] #distancia 
y=dfa[0,1::] #valor real
z=np.array(dfa[1::,1::]).T #valor en pixeles

X,Y=np.meshgrid(x,y)
Z=z
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,cmap='hot')
ax.set_xlabel('Distancia')
ax.set_ylabel('Valor Real')
ax.set_zlabel('Valor en pixeles')
ax.set_title('Plano de la camara')