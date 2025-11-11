import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

def extract_xy_sequence(video_path,
                        min_detect=0.5,
                        min_track=0.5,
                        model_complexity=1):
    
    """
    Extrae la secuencia de 33 puntos clave (landmarks) del cuerpo
    en coordenadas (x, y) normalizadas para cada frame del vídeo.

    Devuelve:
        sequence → array de forma (num_frames, 33, 2)
    """
    

    sequence = []   # Lista donde almacenaremos los landmarks de cada frame

    cap = cv2.VideoCapture(video_path)
    # Abrimos el archivo de vídeo para poder leer frame por frame

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detect,
        min_tracking_confidence=min_track
    ) as pose:
        """
        Crea un objeto "pose" que se encarga de detectar la pose humana
        en cada imagen. Está configurado para:
          - static_image_mode=False → optimiza para vídeo (tracking)
          - model_complexity → control de precisión/velocidad
          - min_detection_confidence → confianza mínima para detectar
          - min_tracking_confidence → confianza mínima para mantener seguimiento
        """

        while True:
            ret, frame = cap.read()
            # ret: indica si se pudo leer el frame
            # frame: imagen actual

            if not ret:
                break   # Salimos si se terminó el vídeo

             # OpenCV usa BGR; MediaPipe necesita RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesa el frame para detectar la pose
            results = pose.process(img)

            # Si se detectaron landmarks de la pose
            if results.pose_landmarks:
                # extraemos 33 puntos: x,y
                pts = []

                # Cada landmarks.landmark contiene x,y,z,visibility
                # Pero solo nos interesan x,y
                for lm in results.pose_landmarks.landmark:
                    pts.extend([lm.x, lm.y])  # <-- SOLO x,y

                pts = np.array(pts)    # 33*2 = 66 elementos

            else:
                # si no detecta → rellenar con ceros
                pts = np.zeros(66)
                print(f"Warning: No se detectaron landmarks en un frame de {video_path}")

            sequence.append(pts)

    cap.release()
    sequence = np.array(sequence)    # (num_frames, 66)
    return sequence


def draw_landmarks_on_frame(frame, landmarks, connections=None, point_color=(0,0,255), line_color=(0,255,0), radius=3, thickness=2, skip_zeros=True):
    """
    Dibuja los landmarks sobre un frame (imagen en BGR).

    Args:
        frame (np.ndarray): imagen BGR donde dibujar (modificada in-place).
        landmarks: array shape (66,) o (num_landmarks, 2) con coordenadas normalizadas (x, y).
                   Puede ser también (num_landmarks*2,) aplanado.
        connections: iterable de pares (a, b) indicando conexiones entre landmarks. Si None,
                     se intentará usar mp_pose.POSE_CONNECTIONS si MediaPipe está importado.
        point_color, line_color: colores BGR para puntos y líneas.
        radius, thickness: tamaño del punto y grosor de línea.
        skip_zeros: si True, no dibuja puntos con coordenadas exactamente (0,0).

    Returns:
        frame con los dibujos (la misma referencia modificada).
    """
    h, w = frame.shape[:2]

    arr = np.asarray(landmarks)
    if arr.ndim == 1:
        # Aseguramos que tenga forma (n_landmarks, 2)
        if arr.size % 2 != 0:
            raise ValueError("El array de landmarks debe tener un número par de elementos (x,y pairs).")
        arr = arr.reshape(-1, 2)

    # Intentamos usar conexiones de MediaPipe si no se pasan
    if connections is None:
        try:
            connections = mp_pose.POSE_CONNECTIONS
        except Exception:
            connections = []

    # Dibujar puntos
    for (x, y) in arr:
        # x, y están normalizados respecto al ancho/alto
        if skip_zeros and x == 0 and y == 0:
            continue
        cx = int(round(x * w))
        cy = int(round(y * h))
        # Evitamos dibujar fuera del frame
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        cv2.circle(frame, (cx, cy), radius, point_color, -1)

    # Dibujar conexiones
    for a, b in connections:
        if a >= arr.shape[0] or b >= arr.shape[0]:
            continue
        x1, y1 = arr[a]
        x2, y2 = arr[b]
        if skip_zeros and ((x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0)):
            continue
        p1 = (int(round(x1 * w)), int(round(y1 * h)))
        p2 = (int(round(x2 * w)), int(round(y2 * h)))
        # Evitamos líneas parcialmente fuera de imagen
        cv2.line(frame, p1, p2, line_color, thickness)

    return frame


def extract_frame_from_video(video_path, frame_idx=None, time_sec=None, as_bgr=True):
    """
    Extrae un único frame de un vídeo.

    Parámetros:
      - video_path (str): ruta al archivo de vídeo.
      - frame_idx (int, opcional): índice del frame (0-based). Si se proporciona, se usa.
      - time_sec (float, opcional): tiempo en segundos para extraer el frame. Si se proporciona y
        frame_idx es None, se usará time_sec.
      - as_bgr (bool): si True devuelve la imagen en BGR (formato OpenCV). Si False devuelve RGB.

    Devuelve:
      - frame (np.ndarray) o None si no se pudo leer.

    Notas:
      - Si ni frame_idx ni time_sec se pasan, se devolverá el primer frame (índice 0).
      - La función maneja bounds y verifica que el índice o tiempo esté en el rango del vídeo.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    # Determinar el frame objetivo
    target_idx = 0
    if frame_idx is not None:
        if frame_idx < 0:
            raise ValueError("frame_idx debe ser >= 0")
        # clamp
        if total_frames > 0:
            target_idx = min(frame_idx, total_frames - 1)
        else:
            target_idx = frame_idx
    elif time_sec is not None:
        if time_sec < 0:
            raise ValueError("time_sec debe ser >= 0")
        if fps > 0:
            target_idx = int(round(time_sec * fps))
            if total_frames > 0:
                target_idx = min(target_idx, total_frames - 1)
        else:
            # si FPS desconocido, fallback a 0
            target_idx = 0
    else:
        target_idx = 0

    # Posicionar y leer
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    if not as_bgr:
        # convertir a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

