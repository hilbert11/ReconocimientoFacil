import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import platform

# --- Sonido ---
if platform.system() == "Windows":
    import winsound
    def beep():
        winsound.Beep(1000, 500)
else:
    def beep():
        print('\a')  # beep simple en consola

# --- Inicializar Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- Funciones EAR / MAR / Pose ---
def eye_aspect_ratio(eye_landmarks, frame_w, frame_h):
    coords = [(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in eye_landmarks]
    A = distance.euclidean(coords[1], coords[5])
    B = distance.euclidean(coords[2], coords[4])
    C = distance.euclidean(coords[0], coords[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear, coords

def mouth_aspect_ratio(mouth_landmarks, frame_w, frame_h):
    coords = [(int(landmark.x * frame_w), int(landmark.y * frame_h)) for landmark in mouth_landmarks]
    A = distance.euclidean(coords[2], coords[3])
    C = distance.euclidean(coords[0], coords[1])
    mar = A / C if C != 0 else 0
    return mar, coords

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

# --- Mostrar alerta interactiva ---
def mostrar_alerta(frame, mensaje="⚠ Pausa activa recomendada!"):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 100), (w-50, 200), (0, 0, 255), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, mensaje, (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.imshow("Deteccion Estrés / Fatiga / Somnolencia", frame)
    beep()
    while True:
        # Detectar cierre con X
        if cv2.getWindowProperty("Deteccion Estrés / Fatiga / Somnolencia", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 27]:  # Enter o ESC
            break

# --- Landmarks ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14, 17]

# --- Parámetros ---
EAR_THRESH = 0.25
CLOSED_FRAMES = 15
MAR_THRESH = 0.6
PITCH_THRESH = 15

VENTANA_TIEMPO = 15  # segundos
UMBRAL_EVENTOS = 100
eventos = []

tiempo_ultimo_chequeo = time.time()

# --- Variables ---
frame_counter = 0
fatiga_detectada = False

# --- Video ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    ahora = time.time()
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Ojos
            left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            left_ear, left_coords = eye_aspect_ratio(left_eye_landmarks, w, h)
            right_ear, right_coords = eye_aspect_ratio(right_eye_landmarks, w, h)
            ear_avg = (left_ear + right_ear) / 2.0
            for (x, y) in left_coords + right_coords:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Boca
            mouth_landmarks = [face_landmarks.landmark[i] for i in MOUTH]
            mar, mouth_coords = mouth_aspect_ratio(mouth_landmarks, w, h)
            for (x, y) in mouth_coords:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Cabeza
            pitch, yaw, roll = head_pose_estimation(face_landmarks, w, h)

            # Detección ojos (fatiga)
            if ear_avg < EAR_THRESH:
                frame_counter += 1
            else:
                frame_counter = 0
                fatiga_detectada = False
            if frame_counter >= CLOSED_FRAMES:
                fatiga_detectada = True

            # Lógica combinada
            estado = "Normal"
            color = (0, 255, 0)

            if fatiga_detectada and mar > MAR_THRESH:
                estado = "Bostezo detectado (Cansancio)"
                color = (255, 0, 0)
                eventos.append(ahora)
            elif fatiga_detectada:
                estado = "Fatiga detectada!"
                color = (0, 0, 255)
                eventos.append(ahora)
            elif mar > MAR_THRESH:
                estado = "Boca abierta (Estrés/Bostezo)"
                color = (0, 255, 255)
                eventos.append(ahora)
            elif pitch > PITCH_THRESH:
                estado = "Cabeceo detectado (Somnolencia)"
                color = (128, 0, 255)
                eventos.append(ahora)

            cv2.putText(frame, estado, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # --- Revisar ventana de tiempo ---
    if ahora - tiempo_ultimo_chequeo >= VENTANA_TIEMPO:
        conteo = sum(1 for t in eventos if t >= tiempo_ultimo_chequeo)
        if conteo >= UMBRAL_EVENTOS:
            mostrar_alerta(frame)
        tiempo_ultimo_chequeo = ahora
        eventos = [t for t in eventos if ahora - t <= VENTANA_TIEMPO]

    cv2.imshow("Deteccion Estrés / Fatiga / Somnolencia", frame)

    # Salir con ESC o con la X
    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Deteccion Estrés / Fatiga / Somnolencia", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
