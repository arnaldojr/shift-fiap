import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dropout, Activation, Input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from os import listdir
import os

# Parâmetros globais
COLOR = (0, 255, 0)
EPSILON = 0.30  # Limiar de similaridade coseno
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Inicializa o classificador cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)  # Normaliza a entrada na escala de [-1, +1]

def load_vgg_face_model(weights_path='vgg_face_weights.h5'):
    input_layer = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((1, 1))(input_layer)
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Convolution2D(4096, (7, 7), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(4096, (1, 1), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(2622, (1, 1))(x)
    x = Flatten()(x)
    out = Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=out)
    model.load_weights(weights_path)

    # Retorna o modelo sem a última camada softmax (extração de features)
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    return feature_model

def find_cosine_similarity(source_representation, test_representation):
    a = np.dot(source_representation, test_representation)
    b = np.linalg.norm(source_representation) * np.linalg.norm(test_representation)
    return 1 - (a / b)

def load_employee_representations(model, employee_pictures_path):
    employees = {}
    for file in listdir(employee_pictures_path):
        try:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                employee, _ = file.split(".")
                employees[employee] = model.predict(preprocess_image(os.path.join(employee_pictures_path, file)))[0, :]
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return employees

def recognize_faces_in_video(video_path, model, employees):
    """Reconhece faces em um vídeo."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])), interpolation=cv2.INTER_AREA)
        faces = face_cascade.detectMultiScale(img, 1.2, 5)

        for (x, y, w, h) in faces:
            if w > 20:
                cv2.rectangle(img, (x, y), (x + w, y + h), COLOR, 4)
                
                detected_face = img[y:y+h, x:x+w]
                detected_face = cv2.resize(detected_face, (224, 224))
                img_pixels = img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels = preprocess_input(img_pixels)

                captured_representation = model.predict(img_pixels)[0, :]
                
                for employee_name, representation in employees.items():
                    similarity = find_cosine_similarity(representation, captured_representation)
                    if similarity < EPSILON:
                        label_name = f"{employee_name} ({similarity:.2f})"
                        cv2.putText(img, label_name, (x + w, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)
                        break

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_vgg_face_model()
    employee_pictures_path = "faces"
    employees = load_employee_representations(model, employee_pictures_path)
    print("Representações de funcionários carregadas com sucesso")
    
    video_path = 1 #'Jeff.mp4'
    recognize_faces_in_video(video_path, model, employees)
