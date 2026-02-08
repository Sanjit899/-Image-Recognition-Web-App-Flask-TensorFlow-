import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load model once
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# ---- COLORS (Dark Mode) ----
BG_COLOR = "#121212"
BTN_COLOR = "#1f1f1f"
BTN_HOVER = "#2a2a2a"
TEXT_COLOR = "#ffffff"
SUBTEXT_COLOR = "#bbbbbb"

# ---- MAIN WINDOW ----
root = tk.Tk()
root.title("Image Recognition App")
root.geometry("520x640")
root.configure(bg=BG_COLOR)

image_path = None

# ---- FUNCTIONS ----
def select_image():
    global image_path

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg *.jfif *.webp")]
    )

    if not file_path:
        return

    image_path = file_path
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((320, 320))
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk
    result_label.config(text="")

def recognize_image():
    if not image_path:
        result_label.config(text="Please select an image first")
        return

    result_label.config(text="Analyzing image...")
    root.update_idletasks()

    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

    text = ""
    for _, name, confidence in decoded[0]:
        text += f"{name}: {confidence*100:.2f}%\n"

    result_label.config(text=text)

# ---- BUTTON HOVER EFFECT ----
def on_enter(e):
    e.widget["background"] = BTN_HOVER

def on_leave(e):
    e.widget["background"] = BTN_COLOR

# ---- UI ELEMENTS ----
title = tk.Label(
    root,
    text="Image Recognition",
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    font=("Segoe UI", 18, "bold")
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Offline AI • No API • Dark Mode",
    bg=BG_COLOR,
    fg=SUBTEXT_COLOR,
    font=("Segoe UI", 10)
)
subtitle.pack()

btn_select = tk.Button(
    root,
    text="Select Image",
    bg=BTN_COLOR,
    fg=TEXT_COLOR,
    font=("Segoe UI", 12),
    activebackground=BTN_HOVER,
    activeforeground=TEXT_COLOR,
    relief="flat",
    padx=20,
    pady=8,
    command=select_image
)
btn_select.pack(pady=15)
btn_select.bind("<Enter>", on_enter)
btn_select.bind("<Leave>", on_leave)

btn_recognize = tk.Button(
    root,
    text="Recognize Image",
    bg=BTN_COLOR,
    fg=TEXT_COLOR,
    font=("Segoe UI", 12),
    activebackground=BTN_HOVER,
    activeforeground=TEXT_COLOR,
    relief="flat",
    padx=20,
    pady=8,
    command=recognize_image
)
btn_recognize.pack(pady=5)
btn_recognize.bind("<Enter>", on_enter)
btn_recognize.bind("<Leave>", on_leave)

img_label = tk.Label(root, bg=BG_COLOR)
img_label.pack(pady=15)

result_label = tk.Label(
    root,
    text="",
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    font=("Segoe UI", 12),
    justify="center"
)
result_label.pack(pady=10)

root.mainloop()
