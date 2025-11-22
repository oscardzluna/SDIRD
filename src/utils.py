import os
import cv2
import datetime


class Utils:

    @staticmethod
    def save_capture(frame, path):
        """
        Guarda una imagen con timestamp en la carpeta de capturas.

        :param frame: Imagen a capturar
        :type frame: np.array

        :param path: Ruta para guardar la captura
        :type path: str

        :return filepath: Ruta de la imagen guardada
        :rtype filepath: str
        """
        os.makedirs(path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.jpg"

        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            cv2.imwrite(filepath, frame)

        print(f"Captura guardada como '{path}/{filename}'.")

        return filepath
