import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import tkinter as tk
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

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
            if selected_language == "Arabic":
                prediction = arabic_model.predict([np.asarray(data_aux)])
            else:
                prediction = english_model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            text_var.set(text_var.get() + predicted_character)
            speak(predicted_character)  # Speak out the predicted character

            time.sleep(1)  # Add a 1 second delay between detections

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    img = img.resize((300, 300))  # Resize without ANTIALIAS
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    panel.after(10, detect_gesture)


def select_language():
    global selected_language
    selected_language = language_var.get()
    if selected_language == "Arabic":
        speak("Arabic sign language selected.")
    else:
        speak("English sign language selected.")


root = tk.Tk()
root.title("Sign Language Translation")

# Create colored background frame
bg_frame = tk.Frame(root, bg="light blue")
bg_frame.pack(fill="both", expand=True)

# Title label
label = tk.Label(bg_frame, text="Sign Language Translation", font=("Helvetica", 16), fg="black", bg="light blue")
label.pack(pady=10)

frame1 = tk.Frame(bg_frame, bg="light blue")
frame1.pack(padx=10, pady=10)

# Dropdown menu for selecting language
languages = ['Arabic', 'English']
language_var = tk.StringVar()
language_var.set(languages[0])  # Set default language to Arabic
language_menu = tk.OptionMenu(frame1, language_var, *languages)
language_menu.pack(side=tk.LEFT)

# Button to select language
select_language_button = tk.Button(frame1, text="Select Language", command=select_language)
select_language_button.pack(side=tk.LEFT)

panel = tk.Label(bg_frame)
panel.pack(padx=10, pady=10)

cap = cv2.VideoCapture(0)

selected_language = "Arabic"  # Default language selection

root.after(10, detect_gesture)
root.mainloop()

cap.release()
cv2.destroyAllWindows()
