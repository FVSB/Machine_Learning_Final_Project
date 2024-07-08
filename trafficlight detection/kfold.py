import os
import numpy as np
from sklearn.model_selection import KFold
import yaml
from ultralytics import YOLO
import shutil
from multiprocessing import freeze_support

current_path = os.getcwd()
image_folder = os.path.join(current_path, 'dataset/images/train')
label_folder = os.path.join(current_path, 'dataset', 'labels', 'train')

# Obtener la lista de archivos de imagen
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Convertir la lista de archivos a un array numpy
X = np.array(image_files)

if __name__ == '__main__':
    freeze_support()
    # Configurar la validación cruzada
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterar sobre los folds
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        train_images = X[train_index]
        val_images = X[val_index]
        
        # Crear carpetas para este fold
        fold_dir = os.path.join(current_path, f'fold_{fold+1}')
        os.makedirs(os.path.join(fold_dir, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'images/val'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', 'val'), exist_ok=True)
        
        for img in train_images:
            shutil.copy2(os.path.join(image_folder, img), os.path.join(fold_dir, 'images', 'train', img))
            label = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(os.path.join(label_folder, label)):
                shutil.copy2(os.path.join(label_folder, label), os.path.join(fold_dir, 'labels', 'train', label))
        for img in val_images:
            shutil.copy2(os.path.join(image_folder, img), os.path.join(fold_dir, 'images', 'val', img))
            label = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(os.path.join(label_folder, label)):
                shutil.copy2(os.path.join(label_folder, label), os.path.join(fold_dir, 'labels', 'val', label))
        
        # Crear un diccionario con la configuración
        # Crear un diccionario con la configuración
        data = {
            'train': f'{fold_dir}/images/train',
            'val': f'{fold_dir}/images/val',
            'nc': 2,  # número de clases
            'names': ['red light', 'green light']
        }

        # Guardar en un archivo YAML
        yaml_path = f'data_fold_{fold+1}.yaml'
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)

        # Guardar las listas de imágenes para este fold
        with open(f'train_images_fold_{fold+1}.txt', 'w') as f:
            f.write('\n'.join(train_images))
        with open(f'val_images_fold_{fold+1}.txt', 'w') as f:
            f.write('\n'.join(val_images))

        # Entrenar el modelo para este fold
        model = YOLO('best.pt')  # Cargar el modelo base
        results = model.train(
            data=yaml_path,
            epochs=15,
            imgsz=640,
            batch=16,
            name=f'yolov8m_traffic_light_fold_{fold+1}',
            device=0
        )

        # Eliminar los enlaces simbólicos después del entrenamiento
        # Eliminar las copias después del entrenamiento
        for img in os.listdir(os.path.join(fold_dir, 'images/train')):
            os.remove(os.path.join(fold_dir, 'images/train', img))
        for img in os.listdir(os.path.join(fold_dir, 'images/val')):
            os.remove(os.path.join(fold_dir, 'images/val', img))
        for label in os.listdir(os.path.join(fold_dir, 'labels/train')):
            os.remove(os.path.join(fold_dir, 'labels/train', label))
        for label in os.listdir(os.path.join(fold_dir, 'labels/val')):
            os.remove(os.path.join(fold_dir, 'labels/val', label))

    print("Entrenamiento con validación cruzada completado.")