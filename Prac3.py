# -*- coding: utf-8 -*-

#PRACTICA 3

import numpy as np
import matplotlib.pyplot as plt 
from skimage import data, io, filters, color
import cv2
from PIL import Image 
from sklearn.cluster import KMeans
import cv2
from skimage.color import rgb2gray
from dtaidistance import dtw


#-------------------------------------------
#SECCION LASER
#------------------------------------------

def labelling(ima): ################################################################################

    ai=np.zeros((ima.shape[0]+2,ima.shape[1]+2)) #se añaden dos bordes a la imagen para no
                                                 #tener tantas condicionales de frontera
    for i in range(ima.shape[0]):
            ai[i+1,1:ai.shape[1]-1]=ima[i,:]
    
    # se realiza el primer ciclo 
    sal1=np.zeros(ai.shape)
    c=2
    
    for i in range (1,ai.shape[0]-1,1):
        for j in range (1,ai.shape[1]-1,1):
            if ai[i,j]==1:
               vec=np.array([sal1[i-1,j-1],sal1[i-1,j],sal1[i-1,j+1],  #aqui preguntamos por los 8 vecinos 
                             sal1[i  ,j-1],            sal1[i  ,j+1],  #del pixel para asignar etiqueta
                             sal1[i+1,j-1],sal1[i+1,j],sal1[i+1,j+1]])
               vec=np.where(vec<=0,1000,vec) #se sustituyen los ceros por un valor muy grande diferente
               #print(vec)
               mn=np.min(vec) #se saca la etiqueta minima de los vecinos si es que hay
               if mn ==1000: 
                   sal1[i,j]=c #si el minimo es igual 100 se asigna una nueva etiqueta
               else:
                   sal1[i,j]=mn #si un vecino ya tiene etiqueta, se asigna la menor 
            else: 
                if ai[i,j-1]!=0: #condicion que permite solo un incremento 
                 c=c+1
                 
    #se realiza el segundo ciclo 
               
    sal2=np.zeros(ai.shape)
    vp=[]
    for i in range (1,ai.shape[0]-1,1):
        for j in range (1,ai.shape[1]-1,1):
            if sal1[i,j]!=0:   #funcion solo aplicamos a los que sean diferente de cero
                vec=np.array([sal1[i-1,j-1],sal1[i-1,j],sal1[i-1,j+1],  #aqui preguntamos por los 8 vecinos 
                              sal1[i  ,j-1],            sal1[i  ,j+1],  #del pixel para asignar etiqueta
                              sal1[i+1,j-1],sal1[i+1,j],sal1[i+1,j+1]])
                for k in range(vec.shape[0]): #preguntamos para cada vecino 
                    if sal1[i,j]!=vec[k] and vec[k]!=0: # si la etiqueta del vecino no es 0 y es diferente
                        sal1=np.where(sal1==vec[k],sal1[i,j],sal1) #se sustituyen las etiquetas los vecinos que coincidan
    
    for i in range (1,ai.shape[0]-1,1):
        for j in range (1,ai.shape[1]-1,1):
            if sal1[i,j]!=0:
                if sal1[i,j] not in vp:
                    vp.append(sal1[i,j])  
    vp=np.array(vp)
    sal1=sal1[1:sal1.shape[0]-1,1:sal1.shape[1]-1]
    #print('sal'+str(vp.shape[0]))
      
    return sal1,vp
#################################################################################################################

def excol(name):
    ima=io.imread(name)
    plt.figure(0)
    plt.title('Da Click el color que deseas aislar y espera...')
    plt.imshow(ima)
    co=np.int32(plt.ginput(1))
    
    #-------------------Modelo de color RGB Kmeans--------------------
    ima33=io.imread(name)
    
    ima33=io.imread(name)
    ima3=color.rgb2luv(ima33)
    
    
    l=(ima3[:,:,0]) #convierte la matriz en un vector
    a=(ima3[:,:,1])
    b=(ima3[:,:,2]) 
    L=l.reshape((-1,1))      
    A=a.reshape((-1,1))
    B=b.reshape((-1,1))
    datos3=np.concatenate((L,A,B),axis=1)
    clases=4
    salida3=KMeans(n_clusters=clases)
    salida3.fit(datos3)
    
    centros3=salida3.cluster_centers_
    aa2=color.lab2rgb(centros3[np.newaxis,:])
    etiquetas3=salida3.labels_ #volver a reconstruir como imagen
    
    for i in range (L.shape[0]): #asignar un color a cada posicion segun la etiqueta
        L[i]=aa2[0][etiquetas3[i]][0]
        A[i]=aa2[0][etiquetas3[i]][1]
        B[i]=aa2[0][etiquetas3[i]][2]
    
    L.shape=l.shape #redimencionar un vector a matriz 
    A.shape=a.shape
    B.shape=b.shape
    
    L=L[:,:,np.newaxis]
    A=A[:,:,np.newaxis]
    B=B[:,:,np.newaxis]
    
    new3=np.concatenate((L,A,B),axis=2)
    gris = rgb2gray(new3)
    bina = np.where(gris == gris[co[0][1],co[0][0]],1,0)
                           
    return bina

def colaser(filf,namec):
    cap = cv2.VideoCapture(namec)
    leido, frame = cap.read()
    if leido == True:
        cv2.imwrite('las.jpg',frame)
        imp=Image.open('las.jpg')
        imp=imp.resize((640,480))
        imp.save('las.jpg')
        print('Foto tomada exitosamente') 
    else:
        print('Error en la captura')
       
    cap.release()   
    imal=io.imread('las.jpg')
    
    rojo=imal[:,:,0]-filf
    filred=filters.gaussian(rojo,sigma=10)
    binario=np.where(filred>10,1,0)
    fil1,col1=np.nonzero(binario)
    cf=((np.max(fil1)-np.min(fil1))/2)+np.min(fil1)
    cc=((np.max(col1)-np.min(col1))/2)+np.min(col1)
    
    return binario, cf,cc, imal

camname=0 #nombre de la camara

    
#como se reescala la imagen para procesamiento mas rapido podemos definir el 
#centroide de la camara como a continuacion

cxc=640/2
cyc=480/2
    
for k in range (1):
    disp=input('presiona tecla para obtener el fondo:')

    for i in range(10):
        cap = cv2.VideoCapture(camname)
        leido, frame = cap.read()
        if leido == True:
            cv2.imwrite('fon'+str(i)+'.jpg',frame)
            imp=Image.open('fon'+str(i)+'.jpg')
            imp=imp.resize((640,480))
            imp.save('fon'+str(i)+'.jpg')
            print('Fondo'+str(i)) 
        else:
            print('Error en la captura')
       
    cap.release()
    
    fondo=np.zeros((480,640,10)) #importante ver las dimenciones de la imagen de la camara 
    for frame in range(10):
        imf=io.imread('fon'+str(frame)+'.jpg')
        fondo[:,:,frame]=imf[:,:,0]
    pf=np.mean(fondo,axis=2) #promedio del fondo
    filf=filters.gaussian(pf,sigma=5) #filtro gausiano al gondo 

    print('Enciende el laser')
    plt.pause(10)

    plt.figure(4)
    plt.ion() #interactive on xD
    ival=colaser(filf,camname)
    icx=ival[2]
    icy=ival[1]
    dis=np.abs(cyc-icy)
    teta=-0.0004*(dis**2)+(0.0627*dis)-(1.161)
    ival[0][int(cyc),int(cxc)]=1
    ival[0][int(icy),int(icx)]=0
    plt.imshow(ival[0],cmap='gray')
    print('Continuamos...')
    fon = dis/np.tan(teta)
    print('La distancia a la pared es de: ',fon)
    plt.pause(2)
################################# captura de imagen ###################################
camname=0 #nombre de la camara
   
disp=input('presiona tecla para tomar la foto:')

cap = cv2.VideoCapture(camname)
leido, frame = cap.read()
if leido == True:
    cv2.imwrite('medicion.jpg',frame)
    imp=Image.open('medicion.jpg')
    imp=imp.resize((640,480))
    imp.save('medicion.jpg')
    print('capturado') 
else:
    print('Error en la captura')
   
cap.release()
   
ima=io.imread('medicion.jpg')
pix=[]
x=5
while (x<10):
    
    plt.figure(0)
    plt.imshow(ima)
    plt.pause(0.5)
    pM=np.int32(plt.ginput(2))

    disd=np.sqrt((pM[0][0]-pM[1][0])**2+(pM[0][1]-pM[1][1])**2)
    pix.append(disd)
    #Nota: para romper el ciclo realiza la medicion en el mismo lugar 
    if disd==0:
        break
    print(disd)
    
    #Dibujando lineas
    cv2.line(ima,(pM[0][0],pM[0][1]),(pM[1][0],pM[1][1]),(255,0,0),2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Ancho = -17.6900330972763 + 0.274823743132625*fon + 0.104568304334793*(pix[0])

Alto = -17.6900330972763 + 0.274823743132625*fon + 0.104568304334793*(pix[1])
print('La distancia a la pared es: ',fon)
print('El ancho del objeto es: ',Ancho)
print('El alto del objeto es: ',Alto)