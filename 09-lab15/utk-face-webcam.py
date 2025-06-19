import cv2
import tensorflow as tf
import numpy as np

# Carregar o modelo salvo
model = tf.keras.models.load_model('model/modelo.h5')


ethnicity_classes = ["Branco", "Negro", "Asiatico", "Indiano", "Outros"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (128, 128))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)  # (1, 128, 128, 3)
    return face_img


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        input_face = preprocess_face(face)

        # Fazer a predição
        age_pred, gender_pred, ethnicity_pred = model.predict(input_face)

        idade = age_pred[0][0]
        genero = 'Masculino' if gender_pred[0][0] < 0.5 else 'Feminino'
        etnia = ethnicity_classes[np.argmax(ethnicity_pred[0])]
        
        # saida
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Idade: {idade:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Genero: {genero}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Etnia: {etnia}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
