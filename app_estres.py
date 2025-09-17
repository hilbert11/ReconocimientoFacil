import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

# --- Landmark predictor ---
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# --- Índices para ojos y boca ---
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [60, 61, 62, 63, 64, 65, 66, 67]

# --- Umbrales ---
EAR_THRESH = 0.25
CLOSED_FRAMES = 15
MAR_THRESH = 0.6
PITCH_THRESH = 15
VENTANA_TIEMPO = 15  # segundos
UMBRAL_EVENTOS = 10

# --- Funciones EAR/MAR ---
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[2], mouth_points[6])
    C = distance.euclidean(mouth_points[0], mouth_points[4])
    return A / C if C != 0 else 0

# --- Video processor ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0
        self.fatiga_detectada = False
        self.eventos = []
        self.tiempo_ultimo_chequeo = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ahora = time.time()

        estado = "Normal"
        color = (0, 255, 0)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])

            # Ojos
            left_eye = points[LEFT_EYE]
            right_eye = points[RIGHT_EYE]
            ear_avg = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # Boca
            mar = mouth_aspect_ratio(points[MOUTH])

            # Fatiga por ojos
            if ear_avg < EAR_THRESH:
                self.frame_counter += 1
            else:
                self.frame_counter = 0
                self.fatiga_detectada = False
            if self.frame_counter >= CLOSED_FRAMES:
                self.fatiga_detectada = True

            # Eventos
            if self.fatiga_detectada and mar > MAR_THRESH:
                estado = "Bostezo detectado (Cansancio)"
                color = (255, 0, 0)
                self.eventos.append(ahora)
            elif self.fatiga_detectada:
                estado = "Fatiga detectada!"
                color = (0, 0, 255)
                self.eventos.append(ahora)
            elif mar > MAR_THRESH:
                estado = "Boca abierta (Estrés/Bostezo)"
                color = (0, 255, 255)
                self.eventos.append(ahora)

        # Revisión ventana de tiempo
        if ahora - self.tiempo_ultimo_chequeo >= VENTANA_TIEMPO:
            conteo = sum(1 for t in self.eventos if t >= self.tiempo_ultimo_chequeo)
            if conteo >= UMBRAL_EVENTOS:
                estado = "⚠ Pausa activa recomendada!"
                color = (0, 0, 255)
            self.tiempo_ultimo_chequeo = ahora
            self.eventos = [t for t in self.eventos if ahora - t <= VENTANA_TIEMPO]

        cv2.putText(img, estado, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return img

# --- Streamlit WebRTC ---
webrtc_streamer(
    key="tesis_dlib",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
