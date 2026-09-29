# SDIRD - Sistema de Detección Inteligente y Reporte de Distribución

SDIRD es una solución de visión artificial basada en **YOLOv8** diseñada para monitorear áreas específicas, detectar la presencia de repartidores y paquetes, y enviar notificaciones automáticas a través de Telegram.

## 🚀 Características

- **Detección en Tiempo Real**: Utiliza modelos YOLOv8 para identificar objetos específicos (actualmente configurado para `box` y `delivery man`).
- **Zona de Alerta Personalizada**: Define un área rectangular en el frame. Si un objeto permanece en esta zona por un tiempo determinado (ej. 10 segundos), se dispara una alerta.
- **Conteo de Objetos**: Realiza un seguimiento y conteo de los objetos detectados en cada frame.
- **Notificaciones Automáticas**: Envía mensajes y capturas de pantalla a un chat de Telegram cuando se detecta una actividad sospechosa o relevante.
- **Gestión de Capturas**: Guarda automáticamente imágenes de las detecciones con marcas de tiempo para auditoría.

## 📁 Estructura del Proyecto

```text
SDIRD/
├── models/               # Modelos YOLO (.pt)
├── src/
│   ├── main.py           # Punto de entrada de la aplicación
│   ├── detection.py      # Lógica de detección y gestión de zonas
│   ├── notifications.py  # Integración con la API de Telegram
│   ├── utils.py          # Funciones auxiliares (guardado de imágenes)
│   └── captures/         # Almacenamiento de capturas de alertas
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

## 🛠️ Instalación
Clonar el repositorio:
```text
git clone <url-del-repositorio>
cd SDIRD
```

Instalar dependencias: Se recomienda usar un entorno virtual:
```
python -m venv venv
.\venv\Scripts\activate  # En Windows
pip install -r requirements.txt
```

Configurar variables de entorno: Para las notificaciones de Telegram, configura las siguientes variables en tu sistema:

- TELEGRAM_BOT_TOKEN: El token proporcionado por @BotFather.
- TELEGRAM_CHAT_ID: El ID del chat donde recibirás las alertas.

## 💻 Uso
Para iniciar el sistema de detección, ejecuta el script principal:
```
python src/main.py
```

Controles:
La ventana de visualización mostrará las detecciones y la "Zona de alerta".
Presiona la tecla ESC para finalizar la ejecución de forma segura.

## ⚙️ Configuración
Puedes ajustar el comportamiento del sistema en src/detection.py:
- Umbral de confianza: Cambia conf_threshold en la inicialización del Detector. 
- Zona de límite: Modifica self.limit_zone = (x1, y1, x2, y2) para ajustar el área de monitoreo. 
- Tiempo de alerta: Ajusta self.alert_duration (en segundos) para definir cuánto tiempo debe estar un objeto en la zona antes de notificar. 
- Clases objetivo: Edita self.target_classes para detectar otros objetos (ej. 'person', 'car', etc.).

### Notas adicionales para el usuario:
1. **Modelo**: En `src/main.py` (línea 5), el código busca `../models/best.pt`. Asegúrate de que tu modelo entrenado tenga ese nombre o actualiza la ruta en el código.
2. **Telegram**: Las notificaciones están actualmente comentadas en `src/detection.py` (línea 204). Deberás descomentarlas para activar el envío de fotos.
3. **Dependencias**: He incluido una estructura estándar de `requirements.txt` basada en tu archivo `.in`.