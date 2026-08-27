import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os

# -------------------- CONFIG -------------------- #

MODEL_PATH = "best.pt"
BASE_SPEED = 60  # Base speed in km/h

# Load YOLO model
model = YOLO(MODEL_PATH)

# -------------------- ANALYSIS FUNCTIONS -------------------- #

def calculate_rqi(frame, results):
    boxes = results[0].boxes
    h, w, _ = frame.shape
    total_area = h * w
    damage_area = 0.0

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            damage_area += (x2 - x1) * (y2 - y1)

    rqi = 1 - (damage_area / total_area)
    rqi = max(0.0, min(1.0, rqi))

    return float(rqi), 0 if boxes is None else len(boxes)


def classify_condition(rqi):
    if rqi >= 0.80:
        return "Good", "Low"
    elif rqi >= 0.50:
        return "Moderate", "Medium"
    else:
        return "Poor", "High"


def recommend_speed(rqi):
    return round(BASE_SPEED * rqi, 1)


# -------------------- IMAGE ANALYSIS -------------------- #

def analyze_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    image = cv2.imread(file_path)
    results = model.predict(image, verbose=False)

    rqi, potholes = calculate_rqi(image, results)
    condition, congestion = classify_condition(rqi)
    speed = recommend_speed(rqi)

    result_text.set(
        f"Potholes Detected: {potholes}\n"
        f"RQI: {rqi:.2f}\n"
        f"Road Condition: {condition}\n"
        f"Congestion Risk: {congestion}\n"
        
    )

    plotted = results[0].plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(plotted)
    img = img.resize((700, 450))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk


# -------------------- VIDEO ANALYSIS -------------------- #

def analyze_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )

    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)

    total_rqi = 0
    total_potholes = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 450))
        results = model.predict(frame, verbose=False)

        rqi, potholes = calculate_rqi(frame, results)
        total_rqi += rqi
        total_potholes += potholes
        frame_count += 1

        plotted = results[0].plot()

        cv2.imshow("Video Analysis - Press Q to Exit", plotted)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_rqi = total_rqi / frame_count
        avg_potholes = total_potholes // frame_count
        condition, congestion = classify_condition(avg_rqi)
        speed = recommend_speed(avg_rqi)

        result_text.set(
            f"Average Potholes: {avg_potholes}\n"
            f"Average RQI: {avg_rqi:.2f}\n"
            f"Road Condition: {condition}\n"
            f"Congestion Risk: {congestion}\n"
            
        )


# -------------------- GUI DESIGN -------------------- #

root = tk.Tk()
root.title("Impact of Potholes on Traffic Flow")
root.geometry("780x820")
root.configure(bg="#111111")

title = tk.Label(
    root,
    text="Impact of Potholes on Traffic Flow",
    font=("Helvetica", 20, "bold"),
    bg="#111111",
    fg="white"
)
title.pack(pady=20)

btn_frame = tk.Frame(root, bg="#111111")
btn_frame.pack(pady=10)

img_btn = tk.Button(
    btn_frame,
    text="Upload Road Image",
    command=analyze_image,
    bg="#1f1f1f",
    fg="white",
    font=("Helvetica", 12),
    width=20,
    height=2
)
img_btn.grid(row=0, column=0, padx=20)

video_btn = tk.Button(
    btn_frame,
    text="Upload Road Video",
    command=analyze_video,
    bg="#1f1f1f",
    fg="white",
    font=("Helvetica", 12),
    width=20,
    height=2
)
video_btn.grid(row=0, column=1, padx=20)

result_text = tk.StringVar()

result_label = tk.Label(
    root,
    textvariable=result_text,
    font=("Helvetica", 14),
    bg="#111111",
    fg="#00ffcc",
    justify="left"
)
result_label.pack(pady=20)

image_label = tk.Label(root, bg="#111111")
image_label.pack(pady=10)

root.mainloop()