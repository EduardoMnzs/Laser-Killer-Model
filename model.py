import cv2
import numpy as np
import tensorflow as tf

# Modelo treinado
model = tf.keras.models.load_model('green_shade_classifier.h5')

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Iintervalos de cores para verde claro e verde escuro em HSV
lower_light_green = np.array([35, 40, 40])
upper_light_green = np.array([85, 255, 255])

lower_dark_green = np.array([25, 40, 40])
upper_dark_green = np.array([35, 255, 255])

# Processamento de imagens capturada
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Redimensiona para o tamanho esperado pelo modelo
    image = image.astype('float32') / 255.0  # Normaliza os valores dos pixels
    image = np.expand_dims(image, axis=0)  # Adiciona dimensão extra para o batch
    return image

while True:
    # Capturar frame da câmera
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame para HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Segmentação de cores para verde claro e verde escuro
    mask_light_green = cv2.inRange(hsv_frame, lower_light_green, upper_light_green)
    mask_dark_green = cv2.inRange(hsv_frame, lower_dark_green, upper_dark_green)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_light_green, mask_dark_green)

    # Encontrar contornos
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filtrar pequenas áreas
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            processed_roi = preprocess_image(roi)
            prediction = model.predict(processed_roi)
            if prediction[0] < 0.5:  # Se for verde claro
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Verde Claro', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:  # Se for verde escuro
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Verde Escuro', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar o frame com os retângulos
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()