import os

import requests


class Notification:

    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    def send_telegram_message(self, text):
        """
        Envía un mensaje de texto a través de Telegram.

        :param text: Mensaje a enviar
        :type text: str

        :return: None
        :rtype: None
        """
        url = f"https://api.telegram.org/bot{self.TOKEN}/sendMessage"
        data = {
            "chat_id": self.CHAT_ID,
            "text": text
        }
        requests.post(url, data=data)

    def send_telegram_photo(self, photo_path, caption=None):
        """
        Envía una foto a través de Telegram.

        :param photo_path: Ruta de la foto a enviar
        :type photo_path: str

        :param caption: Leyenda opcional para la foto
        :type caption: str

        :return: None
        :rtype: None
        """
        url = f"https://api.telegram.org/bot{self.TOKEN}/sendPhoto"
        with open(photo_path, "rb") as img:
            files = {"photo": img}
            data = {"chat_id": self.CHAT_ID}
            if caption:
                data["caption"] = caption

            requests.post(url, data=data, files=files)
