from djitellopy import Tello
import cv2
import mediapipe as mp
import time

# Funzioni per il miglioramento dell'immagine
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness  # Aggiusta la luminosità
    alpha = contrast / 127 + 1  # Aggiusta il contrasto
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def white_balance(image):
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Connetti al drone
tello = Tello()
tello.connect()
print(tello.get_battery())
if tello.get_battery() < 30:
    print("Batteria scarica")
    exit()

# Avvia lo streaming della telecamera
tello.streamon()

# Decollo e alza di 90cm
print("Decollo...")
tello.takeoff()
tello.move_up(90)

# Inizializza MediaPipe per il rilevamento facciale e della mano
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inizializza il rilevatore facciale e delle mani
with mp_face_detection.FaceDetection(min_detection_confidence=0) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:

    try:
        while True:
            # Ottieni il frame dalla camera del Tello
            raw_frame = tello.get_frame_read().frame
            if raw_frame is None:
                print("Errore nel recupero del frame!")
                break

            # Riduci la risoluzione per velocizzare il processo
            frame = cv2.resize(raw_frame, (640, 480))

            # Applicare il miglioramento dell'immagine
            frame = adjust_brightness_contrast(frame, brightness=20, contrast=30)
            frame = white_balance(frame)
            frame = reduce_noise(frame)

            # Converte il frame in RGB (MediaPipe funziona con RGB, non BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Rilevamento facciale
            face_results = face_detection.process(rgb_frame)

            # Rilevamento delle mani
            hand_results = hands.process(rgb_frame)
            # Gestione rilevamento facciale
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Disegna il rettangolo intorno al volto
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Calcola il centro del volto
                    face_center_x = x + w // 2
                    frame_center = iw // 2

                    # Calcola la distanza tra il centro del volto e il centro dell'immagine
                    distance = face_center_x - frame_center

                    if abs(distance) > 10:
                        if distance < 0:
                            print("Rotazione a sinistra")
                            tello.rotate_counter_clockwise(15)
                        elif distance > 0:
                            print("Rotazione a destra")
                            tello.rotate_clockwise(15)
                    else:
                        print("Volto centrato")

            # Gestione rilevamento della mano
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Controlla se la mano è aperta
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                    # Calcola la distanza tra il pollice e la base del palmo (indicativo della mano aperta)
                    if (
                        thumb_tip.y < palm_base.y and
                        index_tip.y < palm_base.y and
                        middle_tip.y < palm_base.y and
                        ring_tip.y < palm_base.y and
                        pinky_tip.y < palm_base.y
                    ):
                        print("Mano aperta rilevata! Atterraggio...")
                        tello.land()
                        tello.streamoff()
                        cv2.destroyAllWindows()
                        exit()
            
            if frame is not None and frame.size > 0:
                cv2.imshow("Tello Camera - Face Tracking", frame)

            # Esci premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Atterraggio...")
                tello.land()
                tello.streamoff()
                cv2.destroyAllWindows()
                exit()

    except KeyboardInterrupt:
        print("Interrotto manualmente")