#Como correr este programa: 
#1 Entrar en la carpeta donde se copio el repositorio yolov7
#2 ejecutar cmd en esa carpeta
#3 colocar python detect_vocales.py y presionar tecla enter


# Importa bibliotecas necesarias
import cv2                          # Para capturar video desde la webcam
import torch                        # Para usar modelos YOLOv7 entrenados con PyTorch
import numpy as np                 # Para manipulación de arrays
import serial                      # Para comunicación serial con Arduino
from models.experimental import attempt_load        # Para cargar el modelo YOLOv7
from utils.general import non_max_suppression, scale_coords  # Para filtrar detecciones
from utils.torch_utils import select_device         # Para elegir entre GPU o CPU

# Función principal que detecta vocales en tiempo real y comunica con Arduino
def detect_webcam(weights_path='weights/best.pt', imgsz=256):
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # Usa GPU si está disponible
    model = attempt_load(weights_path, map_location=device)              # Carga modelo con pesos
    model.eval()                                                         # Establece modo evaluación
    names = model.names                                                  # Obtiene nombres de clases
    print("Clases del modelo:", names)

    vocales = ['A', 'E', 'I', 'O', 'U']                                   # Clases esperadas

    try:
        arduino = serial.Serial('COM6', 9600, timeout=1)                 # Conecta con Arduino en COM6
        print("Conexión serial establecida.")
    except Exception as e:
        print(f"No se pudo conectar al Arduino: {e}")                    # Si falla, termina
        return

    cap = cv2.VideoCapture(0)                                            # Abre webcam (índice 0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()                                          # Lee frame de la cámara
        if not ret:
            print("No se pudo leer frame de la cámara")
            break

        img0 = frame.copy()                                              # Copia para mostrar luego
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)                      # Convierte BGR → RGB
        img = cv2.resize(img, (imgsz, imgsz))                            # Redimensiona al tamaño del modelo
        img = img.transpose((2, 0, 1))                                   # Rearma a formato CHW
        img = np.ascontiguousarray(img)                                  # Asegura datos continuos
        img = torch.from_numpy(img).to(device).float() / 255.0           # Normaliza imagen
        img = img.unsqueeze(0)                                           # Agrega dimensión batch

        with torch.no_grad():                                            # Desactiva gradientes
            pred = model(img)[0]                                         # Ejecuta inferencia
            detections = non_max_suppression(pred, .12, 0.45)[0]         # Filtra con NMS

        vocales_detectadas = set()                                       # Guarda vocales encontradas

        if detections is not None and len(detections):
            detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], img0.shape).round()
            for *xyxy, conf, cls in detections:
                cls_name = names[int(cls)]
                if cls_name in vocales:
                    vocales_detectadas.add(cls_name)                     # Guarda vocal detectada
                    arduino.write(cls_name.encode())                    # Enciende LED (envía vocal mayúscula)
                    label = f'{cls_name} {conf:.2f}'                     # Prepara etiqueta visual
                    xyxy = [int(x.item()) for x in xyxy]                # Convierte a enteros
                    cv2.rectangle(img0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(img0, label, (xyxy[0], xyxy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Apaga LEDs de vocales no detectadas
        for vocal in vocales:
            if vocal not in vocales_detectadas:
                arduino.write(vocal.lower().encode())                   # Envía minúscula para apagar LED

        if not vocales_detectadas:
            cv2.putText(img0, "No se detectó ninguna vocal", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Muestra mensaje si no hay detecciones

        cv2.imshow('Detección Webcam', img0)                             # Muestra el frame procesado
        if cv2.waitKey(1) & 0xFF == 27:                                  # Salir con tecla ESC
            break

    cap.release()                                                        # Libera la cámara
    cv2.destroyAllWindows()                                              # Cierra ventanas
    arduino.close()                                                      # Cierra conexión con Arduino
    print("Conexión serial cerrada.")

# Punto de entrada principal del script
if __name__ == '__main__':
    ruta_pesos = r"C:\Users\Fafa_\yolov7\weights\best.pt"  # Ruta al modelo entrenado
    detect_webcam(ruta_pesos)                              # Ejecuta la función principal

