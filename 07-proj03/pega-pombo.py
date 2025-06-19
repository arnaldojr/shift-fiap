import cv2
import numpy as np


AREA_MIN_POMBO = 100

lower_pombo = np.array([56, 0, 45])
upper_pombo = np.array([179, 142, 212])


cap = cv2.VideoCapture('dados/pombos.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_bgr = frame
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_pombo_roi = cv2.inRange(img_hsv, lower_pombo, upper_pombo)

    # Filtra contornos dos pombos
    contours_pombo, _ = cv2.findContours(mask_pombo_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # vou remover os últimos 3 contornos, que são grandes demais e representam o fundo
    contours_pombo = sorted(contours_pombo, key=cv2.contourArea, reverse=False)[:-3]
    # Filtra contornos com base na área mínima
    contours_filtrados = [cnt for cnt in contours_pombo if cv2.contourArea(cnt) > AREA_MIN_POMBO]

    # Desenha os contornos originais
    img_result = img_bgr.copy()
    cv2.drawContours(img_result, contours_pombo, -1, (0, 255, 0), 2)
    img_result1= img_bgr.copy()
    cv2.drawContours(img_result1, contours_filtrados, -1, (0, 255, 0), 2)

    texto = f"detectados antes: {len(contours_pombo)}, depois: {len(contours_filtrados)}"
    cv2.putText(img_result, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_result1, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("antes", img_result)
    cv2.imshow("depois", img_result1)
    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
