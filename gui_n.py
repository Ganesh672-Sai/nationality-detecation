import tkinter as tk
from tkinter import filedialog
from tkinter import *
import threading
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
import time

# Load the facial expression model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the main window
top = tk.Tk()
top.geometry('800x600')
top.title('Nationality Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Fixing file paths with raw strings or using forward slashes
face = cv2.CascadeClassifier(r"C:/Users/mudug/OneDrive/Desktop/nullclass/nationality detecation/haarcascade_frontalface_default.xml")
model = FacialExpressionModel(r"C:/Users/mudug/OneDrive/Desktop/nullclass/nationality detecation/model_n.json", 
                              r"C:/Users/mudug/OneDrive/Desktop/nullclass/nationality detecation/model.weights.h5")

EMOTIONS_LIST = ["bhartiya", "russian", "italian", "chinese"]

is_running = False

# Function to detect nationality
def Detect(file_path):
    global is_running

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    faces = face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=6)

    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (96, 96))
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            roi_rgb = roi_rgb / 255.0
            roi_input = np.expand_dims(roi_rgb, axis=0)
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi_input))]
            label1.configure(foreground="#011638", text="Nationality is " + pred)
    except Exception as e:
        label1.configure(foreground="#011638", text="Unable to detect")

# Function to start detection loop in a thread
def start_detection(file_path):
    global is_running
    is_running = True

    def detection_loop():
        while is_running:
            Detect(file_path)
            time.sleep(1)  # Delay to simulate processing

    thread = threading.Thread(target=detection_loop)
    thread.start()

# Function to stop detection loop
def stop_detection():
    global is_running
    is_running = False

# Display buttons for start and stop detection
def show_Detect_button(file_path):
    start_b = Button(top, text="Start Detection", command=lambda: start_detection(file_path), padx=10, pady=5)
    start_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    start_b.place(relx=0.7, rely=0.46)

    stop_b = Button(top, text="Stop Detection", command=stop_detection, padx=10, pady=5)
    stop_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    stop_b.place(relx=0.82, rely=0.46)

# Upload image and display it on the UI
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        label1.configure(text='Error uploading image')

# UI Layout: Buttons and Labels
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Nationality Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

# Run the mainloop of tkinter
top.mainloop()
