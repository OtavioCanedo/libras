import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)
hands = mp.solutions.hands.Hands(max_num_hands=1)
model = tf.saved_model.load('savedmodel')

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(0)

word = ''
last_letter = ''

@tf.function
def predict(image):
    image = tf.expand_dims(image, 0)
    predictions = model(image)
    return predictions

while True:
    success, img = cap.read()
    img_flipped = cv2.flip(img, 1)
    frameRGB = cv2.cvtColor(img_flipped,cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img_flipped.shape

    if handsPoints != None:
        for hand in handsPoints:
            x_max = 0
            y_max = 0   
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(img_flipped, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

            try:
                imgCrop = img_flipped[y_min-50:y_max+50,x_min-50:x_max+50]
                imgCrop = cv2.resize(imgCrop,(224,224), interpolation=cv2.INTER_AREA)
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array

                prediction = predict(normalized_image_array)
                indexVal = np.argmax(prediction)

                current_letter = classes[indexVal]

                if current_letter != last_letter:
                    last_letter = current_letter

                if cv2.waitKey(1) & 0xFF == 13:
                    word += current_letter
                    last_letter = current_letter

                if last_letter:
                    cv2.putText(img_flipped,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)

                if word:
                    cv2.putText(img_flipped, word, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            except:
                continue

    elif word:
        cv2.putText(img_flipped, word, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        word += " "

    if key == 8:
        if len(word) > 0:
            word = word[:-1]
    
    if key == 27:  
        break

    cv2.imshow('Imagem', img_flipped)

    if cv2.getWindowProperty('Imagem', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()