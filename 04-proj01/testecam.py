import cv2
import numpy as np
import pyautogui
import time

lower_amarelo = np.array([12, 120, 180])  

upper_amarelo = np.array([18, 180, 255])

lower_azul = np.array([90, 80, 100])  
upper_azul = np.array([115, 200, 255])

area_minima = 10000

def desenha_centro(frame, cx, cy, cor):
    size = 10
    cv2.line(frame,(cx - size,cy),(cx + size,cy),cor,5)
    cv2.line(frame,(cx,cy - size),(cx, cy + size),cor,5)



# Iniciando a captura de vídeo
cap = cv2.VideoCapture(1) ### no meu caso é o 1, mas pode ser 0 ou 2 dependendo da câmera


while True:
    # Tenta fazer a Captura do frame
    ret, frame = cap.read()

    # verifica se o frame foi capturado corretamente
    if not ret:
        print("Erro: Não foi possível capturar o frame.")
        break
    frame = cv2.flip(frame, 1)  # Inverte o frame horizontalmente
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # processa o frame capturado
    mask_amarelo = cv2.inRange(image_hsv, lower_amarelo, upper_amarelo)
    mask_azul = cv2.inRange(image_hsv, lower_azul, upper_azul)
    
    contornos_amarelo, _ = cv2.findContours(mask_amarelo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_azul, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos_amarelo) > 0 and len(contornos_azul) > 0:
        # Encontra o maior contorno
        maior_amarelo = max(contornos_amarelo, key = cv2.contourArea)
        maior_azul = max(contornos_azul, key = cv2.contourArea)
    
        M_am = cv2.moments(maior_amarelo)
        M_az = cv2.moments(maior_azul)
        
        if M_am['m00'] >= area_minima and M_az['m00'] >= area_minima:
            cx_am = int(M_am['m10']/M_am['m00'])
            cy_am = int(M_am['m01']/M_am['m00'])
            cx_az = int(M_az['m10']/M_az['m00'])
            cy_az = int(M_az['m01']/M_az['m00'])
        
            # desenha_centro(frame, cx_am, cy_am, (0, 255, 0))
            # desenha_centro(frame, cx_az, cy_az, (0, 0, 255))
    
        
            cv2.line(frame,(cx_am,cy_am),(cx_az,cy_az),(0,0,255),5)

            angulo = np.arctan2(cy_az - cy_am, cx_az - cx_am)
            angulo_graus = np.degrees(angulo)
            angulo_graus = round(angulo_graus, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0)
            text = M_am['m00']
            origem = (0,150)

            cv2.putText(frame, str(text), origem, font,1,color,2,cv2.LINE_AA)
            origem = (0,50)
            if M_am['m00'] > 100000:
                pyautogui.keyDown('w')
                pyautogui.keyUp('s')
            elif M_am['m00'] < 30000:
                pyautogui.keyDown('s')
                pyautogui.keyUp('w')
            if angulo_graus > 20 :
                # print("Direita")
                cv2.putText(frame, "direita", origem, font,1,color,2,cv2.LINE_AA)
                pyautogui.keyDown('d')
                pyautogui.keyUp('a')
            elif angulo_graus < -20:
                # print("Esquerda")
                cv2.putText(frame, "esquerda", origem, font,1,color,2,cv2.LINE_AA)
                pyautogui.keyDown('a')
                pyautogui.keyUp('d')
            else:
                # print("pardo")
                cv2.putText(frame, "parado", origem, font,1,color,2,cv2.LINE_AA)
                pyautogui.keyUp('a')
                pyautogui.keyUp('d')
                pyautogui.keyUp('w')
                pyautogui.keyUp('s')
    
    
    
    
    
    
    # Exibe o frame processado
    cv2.imshow('frame original', frame)
    # Aguarda 1 ms e verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()