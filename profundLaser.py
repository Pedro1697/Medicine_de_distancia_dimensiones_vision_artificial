import numpy as np
import matplotlib.pyplot as plt 
from skimage import data, io, filters, color
import cv2
from PIL import Image 

#Calibracion profundidad

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
        print('Erro en la captura')
       
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
    

dat=[]

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

    disp=float(input('enciende laser e ingresa la distancia en profundidad:'))

    plt.figure(4)
    plt.ion() #interactive on xD
    ival=colaser(filf,camname)
    icx=ival[2]
    icy=ival[1]
    dis=np.abs(cyc-icy)
    teta=np.arctan(dis/disp)
    
    ival[0][int(cyc),int(cxc)]=1
    ival[0][int(icy),int(icx)]=0
    dat.append([disp,dis,teta])
    plt.imshow(ival[0],cmap='gray')
    print('Volvemos a iniciar')
    plt.pause(2)
    

