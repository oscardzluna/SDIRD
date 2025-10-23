import os
import cv2
import datetime

class Utils:

    @staticmethod
    def save_capture(frame, class_name, path):
        """
        Guarda una imagen con timestamp en la carpeta de capturas.

        :param frame: Imagen a capturar
        :type frame: np.array

        :param class_name: Nombre del objeto detectado
        :type class_name: str

        :param path: Ruta para guardar la captura
        :type path: str

        :return: None
        :rtype: None
        """
        os.makedirs(path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{class_name}_{timestamp}.jpg"

        filepath = os.path.join(path, filename)
        cv2.imwrite(filepath, frame)

        print(f"Captura guardada como '{path}/{filename}'.")
