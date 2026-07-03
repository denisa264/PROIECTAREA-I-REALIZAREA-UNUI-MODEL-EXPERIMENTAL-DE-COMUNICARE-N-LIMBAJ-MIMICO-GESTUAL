import os
import sys
import uuid
import warnings
import unicodedata

# Setari puse inainte de importul Keras/TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import mediapipe as mp
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from Propozitii import construieste_propozitie


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

FOLDER_CURENT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FOLDER_CURENT, "expresii_data")
MODEL_PATH = os.path.join(FOLDER_CURENT, "expresii_model.keras")
EXPRESII_PATH = os.path.join(FOLDER_CURENT, "expresii_list.npy")

NR_SECVENTE = 50
LEN_SECVENTA = 30

DIM_VECTOR = 194
PRAG_DETECTIE = 0.70
NR_CONFIRMARE = 3

LATIME_CAMERA = 1280
INALTIME_CAMERA = 720

os.makedirs(DATA_PATH, exist_ok=True)


def normalizeaza_nume(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.strip().upper().replace(" ", "_")
    return "".join(c for c in text if c.isalnum() or c == "_")


def proceseaza_frame(frame, holistic):
    imagine_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagine_rgb.flags.writeable = False
    rezultate = holistic.process(imagine_rgb)
    imagine_rgb.flags.writeable = True

    imagine = cv2.cvtColor(imagine_rgb, cv2.COLOR_RGB2BGR)
    return imagine, rezultate


def deseneaza_puncte(imagine, rezultate):
   
    elemente = [
        (rezultate.pose_landmarks, mp_holistic.POSE_CONNECTIONS),
        (rezultate.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
        (rezultate.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
    ]

    for landmarks, conexiuni in elemente:
        if landmarks:
            mp_drawing.draw_landmarks(
                imagine, 
                landmarks, 
                conexiuni,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )


def extrage_vector(rezultate):
    
    corp = np.zeros(17 * 4)
    if rezultate.pose_landmarks:
        corp_landmarks = [
            [punct.x, punct.y, punct.z, punct.visibility]
            for i, punct in enumerate(rezultate.pose_landmarks.landmark)
            if i < 17  # Doar pana la brau
        ]
        if corp_landmarks:
            corp = np.array(corp_landmarks).flatten()
            # Pad cu zeros daca nu avem 17 puncte
            if len(corp) < 17 * 4:
                corp = np.pad(corp, (0, 17 * 4 - len(corp)), 'constant')

    # MANA STANGA
    mana_stanga = (
        np.array(
            [
                [punct.x, punct.y, punct.z]
                for punct in rezultate.left_hand_landmarks.landmark
            ]
        ).flatten()
        if rezultate.left_hand_landmarks
        else np.zeros(21 * 3)
    )

     mana_dreapta = (
        np.array(
            [
                [punct.x, punct.y, punct.z]
                for punct in rezultate.right_hand_landmarks.landmark
            ]
        ).flatten()
        if rezultate.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([corp, mana_stanga, mana_dreapta])


def porneste_camera(camera_index):
    # Centralizam pornirea camerei + setarea rezolutiei intr-un singur loc.
    # Folosim DSHOW pe Windows pentru ca initializeaza mult mai rapid camera.
    if os.name == "nt":
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        return camera

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, LATIME_CAMERA)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, INALTIME_CAMERA)

       latime_efectiva = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    inaltime_efectiva = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera ruleaza la {latime_efectiva}x{inaltime_efectiva}.")

    return camera


def pregateste_fereastra(nume_fereastra):
    # Fereastra redimensionabila ca utilizatorul sa o traga la marimea dorita.
    cv2.namedWindow(nume_fereastra, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nume_fereastra, LATIME_CAMERA, INALTIME_CAMERA)


def deseneaza_ghidaj_incadrare(imagine):
    
    h, w, _ = imagine.shape
    marja_x = int(w * 0.08)
    marja_y = int(h * 0.08)
    cv2.rectangle(
        imagine,
        (marja_x, marja_y),
        (w - marja_x, h - marja_y),
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    # Mic plus in centru
    cx, cy = w // 2, h // 2
    cv2.line(imagine, (cx - 12, cy), (cx + 12, cy), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(imagine, (cx, cy - 12), (cx, cy + 12), (80, 80, 80), 1, cv2.LINE_AA)


def numara_secvente(expresie):
    folder = os.path.join(DATA_PATH, expresie)

    if not os.path.exists(folder):
        return 0

    return len(
        [
            nume
            for nume in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, nume)) and nume.isdigit()
        ]
    )


def colecteaza_o_expresie(expresie, camera_index):
    folder_expresie = os.path.join(DATA_PATH, expresie)
    os.makedirs(folder_expresie, exist_ok=True)

    start = numara_secvente(expresie)

    if start >= NR_SECVENTE:
        print(f"{expresie} are deja {NR_SECVENTE} secvente.")
        return

    camera = porneste_camera(camera_index)

    if not camera.isOpened():
        print("Nu pot porni camera.")
        return

    pregateste_fereastra("Expresii - Colectare")

    print()
    print(f"Colectez expresia: {expresie}")
    print(f"Secvente ramase: {NR_SECVENTE - start}")
    print("Apasa q in fereastra video daca vrei sa opresti expresia curenta.")

    oprit = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for nr_secventa in range(start, NR_SECVENTE):
            folder_secventa = os.path.join(folder_expresie, str(nr_secventa))
            os.makedirs(folder_secventa, exist_ok=True)

            for pauza in range(30):
                ok, frame = camera.read()
                if not ok:
                    oprit = True
                    break

                imagine, rezultate = proceseaza_frame(frame, holistic)
                deseneaza_puncte(imagine, rezultate)

                cv2.putText(
                    imagine,
                    f"Pregatire: {expresie}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    imagine,
                    f"Secventa {nr_secventa + 1}/{NR_SECVENTE}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                deseneaza_ghidaj_incadrare(imagine)
                cv2.imshow("Expresii - Colectare", imagine)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    oprit = True
                    break

            if oprit:
                break

            for nr_cadru in range(LEN_SECVENTA):
                ok, frame = camera.read()
                if not ok:
                    oprit = True
                    break

                imagine, rezultate = proceseaza_frame(frame, holistic)
                deseneaza_puncte(imagine, rezultate)

                vector = extrage_vector(rezultate)
                np.save(os.path.join(folder_secventa, str(nr_cadru)), vector)

                cv2.putText(
                    imagine,
                    f"{expresie} | {nr_secventa + 1}/{NR_SECVENTE}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    imagine,
                    f"Cadru {nr_cadru + 1}/{LEN_SECVENTA}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                deseneaza_ghidaj_incadrare(imagine)
                cv2.imshow("Expresii - Colectare", imagine)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    oprit = True
                    break

            if oprit:
                break

    camera.release()
    cv2.destroyAllWindows()

    if oprit:
        print(f"Colectare oprita pentru {expresie}.")
    else:
        print(f"Colectare completa pentru {expresie}.")


def colecteaza_expresii(camera_index):
    print()
    print("COLECTARE EXPRESII")
    print("Scrie expresia in terminal. Exemplu: BUNA")
    print("Cand ai terminat, scrie: GATA")

    while True:
        text = input("\nExpresie de inregistrat: ").strip()

        if text.upper() == "GATA":
            break

        expresie = normalizeaza_nume(text)

        if not expresie:
            print("Nume invalid.")
            continue

        colecteaza_o_expresie(expresie, camera_index)


def incarca_date_antrenare():
    X = []
    y_text = []

    expresii = sorted(
        [
            nume
            for nume in os.listdir(DATA_PATH)
            if os.path.isdir(os.path.join(DATA_PATH, nume))
        ]
    )

    for expresie in expresii:
        folder_expresie = os.path.join(DATA_PATH, expresie)

        for nume_secventa in sorted(os.listdir(folder_expresie), key=lambda x: int(x) if x.isdigit() else -1):
            if not nume_secventa.isdigit():
                continue

            folder_secventa = os.path.join(folder_expresie, nume_secventa)
            cadre = []

            for nr_cadru in range(LEN_SECVENTA):
                fisier = os.path.join(folder_secventa, f"{nr_cadru}.npy")
                if not os.path.exists(fisier):
                    break
                cadre.append(np.load(fisier))

            if len(cadre) == LEN_SECVENTA:
                X.append(cadre)
                y_text.append(expresie)

    expresii_folosite = sorted(set(y_text))

    if len(expresii_folosite) < 2:
        return None, None, None

    coduri = {expresie: index for index, expresie in enumerate(expresii_folosite)}
    y = np.array([coduri[expresie] for expresie in y_text])

    return np.array(X), y, expresii_folosite


def creeaza_model(nr_expresii):
    model = Sequential(
        [
            Input(shape=(LEN_SECVENTA, DIM_VECTOR)),
            LSTM(64, return_sequences=True, activation="tanh"),
            Dropout(0.2),
            LSTM(128, return_sequences=True, activation="tanh"),
            Dropout(0.2),
            LSTM(64, return_sequences=False, activation="tanh"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(nr_expresii, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


def antreneaza_model():
    X, y, expresii = incarca_date_antrenare()

    if X is None:
        print("Ai nevoie de cel putin 2 expresii colectate complet.")
        return

    y_categoric = to_categorical(y, num_classes=len(expresii))

    nr_clase = len(expresii)
    nr_test = max(nr_clase, int(round(len(X) * 0.15)))

    # Verificam ca raman destule pentru antrenare (cel putin 1 per clasa)
    if len(X) - nr_test < nr_clase:
        print()
        print("Prea putine secvente pentru a antrena.")
        print(f"Ai {len(X)} secvente in total pentru {nr_clase} clase.")
        print(f"Ai nevoie de minim {2 * nr_clase} secvente in total (mai bine 20+ per clasa).")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categoric,
        test_size=nr_test,
        random_state=42,
        stratify=y,
    )

    print()
    print("ANTRENARE MODEL")
    print(f"Expresii: {', '.join(expresii)}")
    print(f"Secvente totale: {len(X)}")
    print(f"Antrenare: {len(X_train)} | Test: {len(X_test)}")

    model = creeaza_model(len(expresii))

    callbacks = [
        EarlyStopping(
            monitor="val_categorical_accuracy",
            patience=25,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_categorical_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=120,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)
    np.save(EXPRESII_PATH, np.array(expresii))

    predictii = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictii, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print()
    print(f"Acuratete: {accuracy_score(y_true, y_pred) * 100:.1f}%")
    print(classification_report(y_true, y_pred, target_names=expresii, zero_division=0))
    print(f"Model salvat: {MODEL_PATH}")


def detectie_live(camera_index=0, save_images=False, use_ml_model=True):
    if not use_ml_model:
        print("Pentru expresii este nevoie de modelul antrenat.")
        return

    if not os.path.exists(MODEL_PATH) or not os.path.exists(EXPRESII_PATH):
        print("Nu exista model antrenat.")
        print("Pune collect_mode=True, colecteaza expresii, apoi ruleaza din nou.")
        return

    model = load_model(MODEL_PATH)
    expresii = np.load(EXPRESII_PATH, allow_pickle=True)

    camera = porneste_camera(camera_index)

    if not camera.isOpened():
        print("Nu pot porni camera.")
        return

    pregateste_fereastra("Expresii - Detectie")

    folder_imagini = os.path.join(FOLDER_CURENT, "Output Images")
    if save_images:
        os.makedirs(folder_imagini, exist_ok=True)

    secventa = []
    buffer = []
    propozitie = []
    expresie_adaugata = ""

    print("Detectie live pornita. q = iesire, r = reset")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            imagine, rezultate = proceseaza_frame(frame, holistic)
            deseneaza_puncte(imagine, rezultate)

            vector = extrage_vector(rezultate)
            secventa = (secventa + [vector])[-LEN_SECVENTA:]

            expresie_curenta = ""
            incredere = 0.0

            if len(secventa) == LEN_SECVENTA:
                predictie = model.predict(np.expand_dims(secventa, axis=0), verbose=0)[0]
                index = int(np.argmax(predictie))
                incredere = float(predictie[index])
                expresie_curenta = str(expresii[index])

                if incredere >= PRAG_DETECTIE:
                    buffer = (buffer + [expresie_curenta])[-NR_CONFIRMARE:]

                    if len(buffer) == NR_CONFIRMARE and len(set(buffer)) == 1:
                        expresie_confirmata = buffer[0]

                        if expresie_confirmata == "STERGE":
                            if propozitie:
                                propozitie.pop()
                        elif expresie_confirmata in ("FINAL", "CONFIRMA"):
                            print(f"Propozitie: {construieste_propozitie(propozitie)}")
                        elif expresie_confirmata != "PAUZA":
                            if not propozitie or propozitie[-1] != expresie_confirmata:
                                propozitie.append(expresie_confirmata)
                                expresie_adaugata = expresie_confirmata
                                print(f"Adaugat in propozitie: {expresie_confirmata}")

                        buffer.clear()

            h, w, _ = imagine.shape
            cv2.rectangle(imagine, (0, 0), (w, 100), (25, 25, 25), -1)

            cv2.putText(
                imagine,
                construieste_propozitie(propozitie),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                imagine,
                f"{expresie_curenta} {incredere * 100:.0f}%",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                imagine,
                f"Adaugat: {expresie_adaugata}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (120, 220, 120),
                1,
                cv2.LINE_AA,
            )

            if save_images:
                cv2.imwrite(os.path.join(folder_imagini, f"{uuid.uuid1()}.jpg"), imagine)

            deseneaza_ghidaj_incadrare(imagine)
            cv2.imshow("Expresii - Detectie", imagine)
            tasta = cv2.waitKey(10) & 0xFF

            if tasta == ord("q"):
                break
            if tasta == ord("r"):
                secventa.clear()
                buffer.clear()
                propozitie.clear()

    camera.release()
    cv2.destroyAllWindows()


def run_realtime_expression_detection(
    camera_index=0,
    save_images=False,
    use_ml_model=True,
    collect_mode=False,
):
    if collect_mode:
        colecteaza_expresii(camera_index)
        antreneaza_model()
    else:
        detectie_live(
            camera_index=camera_index,
            save_images=save_images,
            use_ml_model=use_ml_model,
        )


if __name__ == "__main__":
    # collect_mode=True  -> scrii expresia in terminal, colectezi, apoi modelul se antreneaza automat
    # collect_mode=False -> detectie live
    run_realtime_expression_detection(
        camera_index=0,
        save_images=False,
        use_ml_model=True,
        collect_mode=False,
    )
