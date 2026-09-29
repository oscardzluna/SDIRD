import cv2
from detection import Detector

def main():
    detector = Detector(model_path='../models/best.pt', conf_threshold=0.45)
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:/Users/osc_r/Documents/Master/Presentación/REC2025-10-17_0625315.mp4")

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    print("Detección iniciada.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame_annotated = detector.draw_detections(frame, detections)

        # Contar objetos
        counts = detector.count_objects(detections)
        print("Conteo actual:", counts)

        # Revisar zona de alerta
        alert, objects = detector.check_limit_zone(detections, frame)
        if alert:
            print("Objeto(s) dentro de la zona:", objects)

        cv2.imshow("Deteccion Inteligente", frame_annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detección finalizada.")


if __name__ == "__main__":
    main()
