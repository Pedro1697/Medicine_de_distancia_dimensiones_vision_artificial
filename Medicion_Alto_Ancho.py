import numpy as np
import matplotlib.pyplot as plt 
from skimage import data, io, filters, color
import cv2
from PIL import Image 

#Obtencion de pixeles alto y ancho

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
    
        
   