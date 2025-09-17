import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Inicializar Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 17]

# --- Funciones EAR / MAR / Pose ---
def eye_aspect_ratio(eye_landmarks, frame_w, frame_h):
    coords = [(int(l.x * frame_w), int(l.y * frame_h)) for l in eye_landmarks]
    A = distance.euclidean(coords[1], coords[5])
    B = distance.euclidean(coords[2], coords[4])
    C = distance.euclidean(coords[0], coords[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def mouth_aspect_ratio(mouth_landmarks, frame_w, frame_h):
    coords = [(int(l.x * frame_w), int(l.y * frame_h)) for l in mouth_landmarks]
    A = distance.euclidean(coords[2], coords[3])
    C = distance.euclidean(coords[0], coords[1])
    return A / C if C != 0 else 0

def head_pose_estimation(face_landmarks, frame_w, frame_h):
    image_points = np.array([
        (face_landmarks.landmark[1].x * frame_w, face_landmarks.landmark[1].y * frame_h),
        (face_landmarks.landmark[152].x * frame_w, face_landmarks.landmark[152].y * frame_h),
        (face_landmarks.landmark[33].x * frame_w, face_landmarks.landmark[33].y * frame_h),
        (face_landmarks.landmark[263].x * frame_w, face_landmarks.landmark[263].y * frame_h),
        (face_landmarks.landmark[61].x * frame_w, face_landmarks.landmark[61].y * frame_h),
        (face_landmarks.landmark[291].x * frame_w, face_landmarks.landmark[291].y * frame_h)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [angle[0] for angle in euler_angles]
    return pitch, yaw, roll

# --- Umbrales ---
EAR_THRESH = 0.25
CLOSED_FRAMES = 15
MAR_THRESH = 0.6
PITCH_THRESH = 15
VENTANA_TIEMPO = 15  # segundos
UMBRAL_EVENTOS = 10

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.frame_counter = 0
        self.fatiga_detectada = False
        self.eventos = []
        self.tiempo_ultimo_chequeo = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        ahora = time.time()

        estado = "Normal"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Ojos
                left_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in LEFT_EYE], w, h)
                right_ear = eye_aspect_ratio([face_landmarks.landmark[i] for i in RIGHT_EYE], w, h)
                ear_avg = (left_ear + right_ear) / 2.0

                # Boca
                mar = mouth_aspect_ratio([face_landmarks.landmark[i] for i in MOUTH], w, h)

                # Cabeza
                pitch, yaw, roll = head_pose_estimation(face_landmarks, w, h)

                # Fatiga por ojos
                if ear_avg < EAR_THRESH:
                    self.frame_counter += 1
                else:
                    self.frame_counter = 0
                    self.fatiga_detectada = False
                if self.frame_counter >= CLOSED_FRAMES:
                    self.fatiga_detectada = True

                # Lógica de eventos
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
                elif pitch > PITCH_THRESH:
                    estado = "Cabeceo detectado (Somnolencia)"
                    color = (128, 0, 255)
                    self.eventos.append(ahora)

        # Revisión de ventana de tiempo
        if ahora - self.tiempo_ultimo_chequeo >= VENTANA_TIEMPO:
            conteo = sum(1 for t in self.eventos if t >= self.tiempo_ultimo_chequeo)
            if conteo >= UMBRAL_EVENTOS:
                estado = "⚠ Pausa activa recomendada!"
                color = (0, 0, 255)
            self.tiempo_ultimo_chequeo = ahora
            self.eventos = [t for t in self.eventos if ahora - t <= VENTANA_TIEMPO]

        cv2.putText(img, estado, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return img

# --- Streamlit-WebRTC ---
webrtc_streamer(
    key="tesis",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
