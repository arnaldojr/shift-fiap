##### Implemente seu código aqui......
import cv2
import numpy as np

# abrir video e capturar o primeiro frame
cap = cv2.VideoCapture('lab_images/people-walking.mp4')

ret, frame = cap.read()# criar o objeto de subtração de fundo

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,        # número de frames usados para modelar o fundo
    varThreshold=16,    # limiar de variância
    detectShadows=False
    )
while(1):    
    ret, frame = cap.read() 
    # ler o frame    
    if not ret:        
        break    
    fgmask = fgbg.apply(frame)    # identificar pessoas em movimento    # aplicar morfologia  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
    # kernel = np.ones((5,5), np.uint8)    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)    # encontrar contornos na imagem  
    
    #dilatação
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)
      
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # use a função cv2.boundingRect para obter o retângulo delimitador de cada contorno        
    if len(contours) > 0:        
        max_contorno = max(contours, key=cv2.contourArea)        
        (x1,y1,x2,y2)=cv2.boundingRect(max_contorno)        
        cv2.rectangle(frame,(x1,y1),(x1+x2,y1+y2),(132,235,255),2)        
        cv2.rectangle(fgmask,(x1,y1),(x1+x2,y1+y2),(132,235,255),2)        # exibir a imagem    
    
    cv2.imshow('frame', frame)    # 
    cv2.imshow('fgmask', fgmask)    
    k = cv2.waitKey(30) & 0xff    
    if k == 27:        
        break
cap.release()
cv2.destroyAllWindows()  