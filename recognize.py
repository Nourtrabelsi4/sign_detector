import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'sign', 1: '  ', 2: 'detector'}

engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def detect_gesture():
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if data_aux:
            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            text_var.set(text_var.get() + predicted_character)
            speak(predicted_character)  # Speak out the predicted character

            time.sleep(3)  # Add a 1 second delay between detections

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    img = img.resize((300, 300))  # Resize without ANTIALIAS
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    panel.after(10, detect_gesture)


def translate_text():
    message = message_var.get()
    translated_text = ""
    for char in message:
        if char.upper() in labels_dict.values():
            translated_text += char.upper()  # Translate only if the character is in the dictionary
    text_var.set(translated_text)


root = tk.Tk()
root.title("Sign Language Translation")

# Define colors
bg_color = "#ADD8E6"  # Light gray
title_color = "#333333"  # Dark gray
text_color = "#000000"  # Black

# Create colored background frame
bg_frame = tk.Frame(root, bg=bg_color)
bg_frame.pack(fill="both", expand=True)

# Title label
label = tk.Label(bg_frame, text="Sign Language Translation", font=("Helvetica", 16), fg=title_color, bg=bg_color)
label.pack(pady=10)

frame1 = tk.Frame(bg_frame, bg=bg_color)
frame1.pack(padx=10, pady=10)

# Entry widget for typing messages
message_var = tk.StringVar()
message_entry = tk.Entry(frame1, textvariable=message_var, font=("Helvetica", 14), fg=text_color)
message_entry.pack(side=tk.LEFT)

# Button to translate typed text
translate_button = tk.Button(frame1, text="Translate", command=translate_text)
translate_button.pack(side=tk.LEFT)

# Entry widget for displaying translated text
text_var = tk.StringVar()
text_box = tk.Entry(bg_frame, textvariable=text_var, font=("Helvetica", 14), fg=text_color, bg="white")
text_box.pack(padx=10, pady=10)

panel = tk.Label(bg_frame)
panel.pack(padx=10, pady=10)

#Load and display the image of all signs and letters
image_path = "./signes.png"
img = Image.open(image_path)
img = img.resize((300, 300))  # Resize without ANTIALIAS
photo = ImageTk.PhotoImage(img)
signs_label = tk.Label(bg_frame, image=photo, bg=bg_color)
signs_label.pack(side=tk.RIGHT, padx=10, pady=10)
cap = cv2.VideoCapture(0)

root.after(10, detect_gesture)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
