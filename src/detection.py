import cv2
import time
import datetime
from ultralytics import YOLO
from utils import Utils
from notifications import Notification


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
        self.target_classes = ['amazon', 'amazon logo', 'delivery man', 'dhl', 'estafeta', 'mercado libre', 'package']

        # Identificador de colores por objeto
        self.class_colors = {
            'amazon': (44, 174, 243), # Azul claro
            'amazon logo': (255, 161, 0), # Naranja
            'delivery man': (72, 207, 60), # Verde
            'dhl': (113, 72, 19), # Rosa
            'estafeta': (255, 0, 0), # Rojo
            'mercado libre': (15, 50, 255), # Azul fuerte
            'package': (255, 0, 255), # Rosa
        }

        # Coordenadas de la zona límite (x1, y1, x2, y2)
        self.limit_zone = (150, 100, 850, 650)

        # Bandera para controlar tiempos dentro de la zona
        self.condition_start_time = None

        # Bandera para controlar alertas disparadas
        self.alert_sent = False

        # Tiempo de objeto en la zona para disparar alerta (s)
        self.alert_duration = 10

        # Ruta para capturas de pantalla
        self.path = "captures"

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

                if class_name in self.target_classes and conf >= self.conf_threshold:
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
        detections_in_zone = []
        detected_objects = set()
        alert_trigger = False

        # Detección de objetos dentro de la zona
        for det in detections:
            cls = det['class']
            x1, y1, x2, y2 = det['bbox']

            if x1 < x2_zone and x2 > x1_zone and y1 < y2_zone and y2 > y1_zone:
                detections_in_zone.append(det)
                detected_objects.add(cls)

        has_delivery_man = 'delivery man' in detected_objects
        companion_classes = {'amazon', 'amazon logo', 'dhl', 'estafeta', 'mercado libre', 'package'}
        has_companion = bool(detected_objects.intersection(companion_classes))

        condition_met = has_delivery_man and has_companion
        if condition_met:
            # Iniciar temporizador si se encontraron objetos en la zona
            if self.condition_start_time is None:
                self.condition_start_time = time.time()

            # Calcular tiempo transcurrido
            elapsed = time.time() - self.condition_start_time

            # Verificar si el objeto permanece en la zona
            if elapsed >= self.alert_duration:
                # Solo disparar si no ha sido disparada antes
                if not self.alert_sent:
                    alert_trigger = True
                    print(f"Repartidor dentro de la zona por {int(elapsed)}s")

                    # Ejecutar acciones
                    self.trigger_actions(frame)

                    # Marcar alerta como enviada
                    self.alert_sent = True
        else:
            # Eliminar temporizador si ya no hay objetos en la zona
            self.condition_start_time = None
            self.alert_sent = False

        return alert_trigger, detections_in_zone


    def trigger_actions(self, frame):
        """
        Ejecuta las acciones al disparar la alerta.

        :param frame: Imagen capturada
        :type frame: np.array

        :return: None
        :rtype: None
        """
        filepath = Utils.save_capture(frame, self.path)
        notification = Notification()
        timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
        notification.send_telegram_photo(filepath, f"{timestamp} Un repartidor ha llegado a tu domicilio.")
