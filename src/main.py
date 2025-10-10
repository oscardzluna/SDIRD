from ultralytics import YOLO
import cv2

def main():
    # Carga el modelo YOLOv8 preentrenado (versión ligera)
    model = YOLO('../models/yolov8n.pt')

    # Inicializa la cámara (0 = cámara integrada)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    print("Cámara iniciada. Presiona 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame.")
            break

        # Realiza detección
        results = model(frame, stream=True)

        # Muestra resultados en tiempo real
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Detección de Repartidores", annotated_frame)

        # Salir con tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detección finalizada.")

if __name__ == "__main__":
    main()
