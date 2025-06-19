# Detecção de Face e Olhos em Tempo Real
import cv2
import os

# Carregando os classificadores
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Verificando se os classificadores foram carregados corretamente
if face_cascade.empty() or eye_cascade.empty():
    print("Erro ao carregar os classificadores. Verifique os caminhos!")
    exit()


cap = cv2.VideoCapture(1)

# Verificando se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Não foi possível abrir a câmera!")
    exit()

print("Detecção iniciada! Pressione 'q' para sair.")

while True:
    # Capturando frame por frame
    ret, frame = cap.read()
    
    if not ret:
        print("Não foi possível receber o frame. Saindo...")
        break
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectando faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    # Desenha retângulos nas faces e detecta olhos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Região de interesse para os olhos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detectando olhos
        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor = 1.1, minNeighbors = 11)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Detectando sorisso
        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor = 1.1, minNeighbors = 11)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    
    # Exibindo o resultado
    cv2.imshow('Detecção de Face e Olhos', frame)
    
    # Saindo se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberando os recursos
cap.release()
cv2.destroyAllWindows()