import cv2
import numpy as np
from sort import Sort

import os
import tempfile
import urllib.request

# Data processing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from tqdm import tqdm  # Barra de carga
from IPython.display import clear_output # Limpiar la salida de la celda
from time import sleep  # Importar sleep

from ultralytics import YOLO
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

# Modelos
yolo_model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def apply_model_to_video(video_url, output_video_path, heatmap_path, last_frame_path):    
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    decay = 0.01
    heatmap = np.zeros((h, w), dtype=np.float32)
    tracker = Sort()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Frames")
    grid_size = 50
    data = []
    last_saved_second = -1
    
    last_frame = None  # Para guardar el último frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_seconds = frame_number / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        results = yolo_model.track(frame, persist=True)

        for detection in results:
            class_id = None
            if (detection.boxes is not None) and (len(detection.boxes) > 0):
                class_id = int(detection.boxes.cls[0])

            if class_id == None or yolo_model.names[class_id] != "person":
                continue

            if yolo_model.names[class_id] != "person":
                continue

            box = detection.boxes.xyxy.cpu().numpy().astype(int)
            tracks = tracker.update(box).astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                cv2.putText(frame, f"ID:{track_id}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                grid_x = (xmin + xmax) // 2 // grid_size
                grid_y = (ymin + ymax) // 2 // grid_size

                current_second = int(total_seconds)
                if current_second != last_saved_second:
                    data.append({
                        "ID": track_id,
                        "cuadrante_x": grid_x,
                        "cuadrante_y": grid_y,
                        "timestamp": timestamp
                    })
                    last_saved_second = current_second

                heatmap[ymin:ymax, xmin:xmax] += 1

        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        for x in range(0, w, grid_size):
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
        for y in range(0, h, grid_size):
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
        
        last_frame = overlay.copy()

        clear_output(wait=True)
        out.write(overlay)
        pbar.update(1)

    df = pd.DataFrame(data)
    cap.release()
    out.release()

    create_heatmap(df, heatmap_path)
    cv2.imwrite(last_frame_path, last_frame)

def create_heatmap(df, heatmap_path):
    heatmap = df.groupby(["cuadrante_x", "cuadrante_y"]).size().unstack(fill_value=0)
    heatmap = heatmap.div(heatmap.max().max())
    heatmap = rotate(heatmap, 270)
    
    plt.figure(figsize=(12, 8))  # Adjusting the figure size
    ax = sns.heatmap(heatmap, cmap="viridis", cbar_kws={'label': 'Frecuencia'})

    # Set axis labels and title
    ax.set_xlabel("Cuadrante X")
    ax.set_ylabel("Cuadrante Y")
    plt.title("Heatmap de Frecuencia de Personas en el Video")

    # Save the heatmap as an image
    plt.savefig(heatmap_path)
