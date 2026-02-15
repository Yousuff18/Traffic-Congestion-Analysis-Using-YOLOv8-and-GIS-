import cv2
import csv
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk

# ================= CONFIG =================
VIDEO_PATH = "parking.mp4"
OUTPUT_VIDEO = "parking_detected.mp4"
CSV_FILE = "traffic_live.csv"

HIGH_THRESHOLD = 4
MEDIUM_THRESHOLD = 3

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "person"]
# =========================================


# ================= GLOBAL STATE =================
running = False
cap = None
out = None
model = YOLO("yolov8n.pt")

current_video_second = 0
frame_vehicle_sum = 0
frame_count_in_second = 0
per_class_sum = {k: 0 for k in VEHICLE_CLASSES}

csv_file = None
writer = None
# ===============================================


# ================= UI =====================
root = tk.Tk()
root.title("AI Traffic Monitor")
root.state("zoomed")
root.configure(bg="#0f1115")

container = tk.Frame(root, bg="#0f1115")
container.pack(expand=True)

title = tk.Label(
    container,
    text="AI TRAFFIC MONITOR",
    font=("Segoe UI", 30, "bold"),
    fg="white",
    bg="#0f1115"
)
title.pack(pady=(30, 5))

subtitle = tk.Label(
    container,
    text="Real-time vehicle density analysis using Computer Vision",
    font=("Segoe UI", 14),
    fg="#aaaaaa",
    bg="#0f1115"
)
subtitle.pack(pady=(0, 25))

panel = tk.Frame(container, bg="#0f1115")
panel.pack()

def make_label(text):
    return tk.Label(
        panel,
        text=text,
        font=("Segoe UI", 18),
        fg="white",
        bg="#0f1115",
        anchor="w"
    )

time_label = make_label("Time: 0 s")
time_label.grid(row=0, column=0, sticky="w", pady=4)

labels = {}
row = 1
for cls in VEHICLE_CLASSES:
    lbl = make_label(f"{cls.capitalize()}: 0")
    lbl.grid(row=row, column=0, sticky="w", pady=3)
    labels[cls] = lbl
    row += 1

total_label = make_label("Avg vehicles / sec: 0")
total_label.grid(row=row, column=0, sticky="w", pady=(10, 20))

status_label = tk.Label(
    panel,
    text="LOW",
    font=("Segoe UI", 26, "bold"),
    fg="#43a047",
    bg="#0f1115"
)
status_label.grid(row=row + 1, column=0, pady=(10, 5))

bar_canvas = tk.Canvas(
    panel,
    width=520,
    height=28,
    bg="#1f232b",
    highlightthickness=0
)
bar_canvas.grid(row=row + 2, column=0, pady=(0, 30))
bar = bar_canvas.create_rectangle(0, 0, 520, 28, fill="#43a047")

run_button = ttk.Button(
    container,
    text="▶  RUN ANALYSIS",
    width=28
)
run_button.pack(pady=20)


# ================= LOGIC ==================

def start_analysis():
    global running, cap, out, csv_file, writer
    global current_video_second, frame_vehicle_sum, frame_count_in_second, per_class_sum

    if running:
        return

    running = True
    run_button.state(["disabled"])
    run_button.config(text="RUNNING...")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Video file could not be opened")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    csv_file = open(CSV_FILE, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_sec",
        "avg_vehicles_per_sec",
        "traffic_state",
        "traffic_value"
    ])

    current_video_second = 0
    frame_vehicle_sum = 0
    frame_count_in_second = 0
    per_class_sum = {k: 0 for k in VEHICLE_CLASSES}

    update_loop()


def update_loop():
    global current_video_second, frame_vehicle_sum, frame_count_in_second, per_class_sum

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        csv_file.close()
        print("✅ Analysis finished. Video saved:", OUTPUT_VIDEO)
        return

    results = model.predict(frame, conf=0.3, verbose=False)

    vehicles_this_frame = 0
    per_frame_class = {k: 0 for k in VEHICLE_CLASSES}

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            name = r.names[int(cls)]
            if name in VEHICLE_CLASSES:
                vehicles_this_frame += 1
                per_frame_class[name] += 1

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, name,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    out.write(frame)

    frame_vehicle_sum += vehicles_this_frame
    frame_count_in_second += 1
    for k in VEHICLE_CLASSES:
        per_class_sum[k] += per_frame_class[k]

    video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    video_second = int(video_time)

    if video_second > current_video_second:
        avg_vehicles = int(round(frame_vehicle_sum / max(frame_count_in_second, 1)))

        if avg_vehicles > HIGH_THRESHOLD:
            state, value, color = "HIGH", 3, "#e53935"
        elif avg_vehicles >= MEDIUM_THRESHOLD:
            state, value, color = "MEDIUM", 2, "#fb8c00"
        else:
            state, value, color = "LOW", 1, "#43a047"

        time_label.config(text=f"Time: {video_second} s")

        for k in VEHICLE_CLASSES:
            labels[k].config(
                text=f"{k.capitalize()}: {int(round(per_class_sum[k] / frame_count_in_second))}"
            )

        total_label.config(text=f"Avg vehicles / sec: {avg_vehicles}")
        status_label.config(text=state, fg=color)
        bar_canvas.itemconfig(bar, fill=color)

        writer.writerow([video_second, avg_vehicles, state, value])

        current_video_second = video_second
        frame_vehicle_sum = 0
        frame_count_in_second = 0
        per_class_sum = {k: 0 for k in VEHICLE_CLASSES}

    root.after(1, update_loop)


run_button.config(command=start_analysis)
root.mainloop()
