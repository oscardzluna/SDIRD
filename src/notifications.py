import requests


class Notification:

    TOKEN = "8490805389:AAF1MM7GbPWadvpQ7K4tgp_4Mqx4i1voKag"
    CHAT_ID = "1138150600"

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
