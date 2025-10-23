import cv2
import time
from ultralytics import YOLO
from utils import Utils


class Detector:
    def __init__(self, model_path, conf_threshold):
        """
        Inicializa el detector con el modelo pre-cargado.

        :param model_path: Ruta del modelo
        :type model_path: str

        :param conf_threshold: Umbral de confianza
        :type conf_threshold: float

        :return: None
        :rtype: None
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Objetos
        self.target_classes = ['person', 'motorcycle', 'car', 'truck']

        # Identificador de colores por objeto
        self.class_colors = {
            'person': (0, 255, 0),  # Verde
            'motorcycle': (255, 165, 0),  # Naranja
            'car': (255, 0, 0),  # Azul
            'truck': (0, 0, 255),  # Rojo
        }

        # Coordenadas de la zona límite (x1, y1, x2, y2)
        self.limit_zone = (50, 50, 550, 480)

        # Diccionario para controlar tiempos dentro de la zona
        self.presence_timers = {}

        # Tiempo de objeto en la zona para disparar alerta (s)
        self.alert_duration = 60

        # Ruta para capturas de pantalla
        self.path = "captures"

        # Control de última captura
        self.last_capture_time = dict.fromkeys(self.target_classes, 0)

    def detect(self, frame):
        """
        Ejecuta la detección sobre un frame y devuelve resultados filtrados.

        :param frame: Imagen capturada
        :type frame: np.array

        :return: Lista de objetos detectados
        :rtype: list
        """
        results = self.model(frame, stream=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]

                # Filtra solo los objetivos con la confianza mínima
                if class_name in self.target_classes and conf >= self.conf_threshold:
                    # Coordenadas del bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf, 2),
                        'bbox': (x1, y1, x2, y2)
                    })

        return detections

    def draw_detections(self, frame, detections):
        """
        Dibuja la zona límite y las detecciones sobre la imagen.

        :param frame: Imagen de entrada
        :type frame: np.array

        :param detections: Lista de objetos detectados
        :type detections: list

        :return: Imagen de salida
        :rtype: np.array
        """
        # Zona límite
        x1, y1, x2, y2 = self.limit_zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Zona de alerta", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Objetos
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} ({det['confidence']})"
            color = self.class_colors.get(det['class'], (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def count_objects(self, detections):
        """
        Cuenta cuántos objetos hay en imagen.

        :param detections: Lista de objetos detectados
        :type detections: list

        :return: Diccionario con conteo por objeto
        :rtype: dict
        """
        counts = dict.fromkeys(self.target_classes, 0)
        for det in detections:
            counts[det['class']] += 1

        return counts

    def check_limit_zone(self, detections, frame):
        """
        Verifica si hay alguna detección dentro de la zona de alerta.

        :param detections: Lista de objetos detectados
        :type detections: list

        :param frame: Imagen de entrada
        :type frame: np.array

        :return: Alerta activada y lista de objetos en zona
        :rtype: tuple(bool, list)
        """
        x1_zone, y1_zone, x2_zone, y2_zone = self.limit_zone
        current_time = time.time()
        alert_triggered = False

        alert_objects = []

        # Objetos detectados dentro de la zona
        current_in_zone = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = det['class']

            # Comprobar si el centro del objeto cae dentro del área
            if x1_zone <= cx <= x2_zone and y1_zone <= cy <= y2_zone:
                alert_triggered = True
                alert_objects.append(class_name)
                current_in_zone.append(class_name)

                # Iniciar temporizador si no existía
                if class_name not in self.presence_timers:
                    self.presence_timers[class_name] = current_time

                # Calcular tiempo dentro
                elapsed = current_time - self.presence_timers[class_name]

                # Si supera el umbral y no se ha guardado recientemente
                if (elapsed >= self.alert_duration and
                        current_time - self.last_capture_time[class_name] > self.alert_duration):
                    print(f"{class_name} dentro de la zona por {int(elapsed)}s")
                    Utils.save_capture(frame, class_name, self.path)
                    self.last_capture_time[class_name] = current_time
            else:
                # Si sale de la zona, eliminar el temporizador
                if class_name in self.presence_timers:
                    del self.presence_timers[class_name]

        # Limpiar timers de objetos que ya no están en zona
        for cls in list(self.presence_timers.keys()):
            if cls not in current_in_zone:
                del self.presence_timers[cls]

        return alert_triggered, alert_objects
