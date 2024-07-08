from ultralytics import YOLO
from multiprocessing import freeze_support
import os

# Cargar el modelo pre-entrenado YOLOv8m
model_path = os.path.join(os.path.dirname(__file__), 'yolov8m.pt')
model = YOLO(model_path)

# Realizar fine-tuning
if __name__ == '__main__':
    freeze_support()
    results = model.train(
        data='C:/Users/Diana Laura/Desktop/ML/TrafficLight.yaml',
        epochs=80,
        imgsz=640,
        batch=16,
        name='yolov8m_traffic_light',
        patience=20,
        device=0
    )

    # Guardar el modelo entrenado
    model.save('yolov8m_TrafficLight.pt')