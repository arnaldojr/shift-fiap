import cv2

import numpy as np
#rtsp = "rtsp://ip:porta/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
# Iniciando a captura de vídeo
cap = cv2.VideoCapture(1)  ## 0 para a câmera padrão, 1,3,5 para a câmera secundária
                            ## "video.mp4" para um arquivo de vídeo

while True:
    # Tenta fazer a Captura do frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
  
    # verifica se o frame foi capturado corretamente
    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break
    
    ###### processa o frame capturado
    
    # Converte o frame para escala de cinza
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
                            # [179 168 245]
    image_lower_hsv = np.array([170, 150, 200])  
    image_upper_hsv = np.array([180, 255, 255])


                                # [1 159 221]
    image_lower_hsv2 = np.array([0, 150, 200])  
    image_upper_hsv2 = np.array([15, 255, 255])

    # Cria a mascara
    mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)
    mask_hsv2 = cv2.inRange(img_hsv, image_lower_hsv2, image_upper_hsv2)


    # juntando as duas mascaras
    resultado = cv2.bitwise_or(mask_hsv, mask_hsv2)
    res = cv2.bitwise_and(frame,frame, mask=resultado)

    contornos_simple, _ = cv2.findContours(resultado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contornos_simple) > 0:
        maior_contorno = max(contornos_simple, key = cv2.contourArea)
        cv2.drawContours(frame, maior_contorno, -1, (0, 255, 0), 10)



    
    
    ####### Exibe o frame processado
    cv2.imshow('frame', frame)
    
    # Aguarda 1 ms e verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()