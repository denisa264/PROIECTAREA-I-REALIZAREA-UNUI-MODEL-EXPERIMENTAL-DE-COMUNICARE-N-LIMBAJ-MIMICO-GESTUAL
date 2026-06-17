import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

COLLECT_KEYS_TO_LABELS = {
    ord("a"): "A",
    ord("b"): "B",
    ord("c"): "C",
    ord("d"): "D",
    ord("e"): "E",
    ord("f"): "F",
    ord("g"): "G",
    ord("h"): "H",
    ord("i"): "I",
   # ord("j"): "J",
    ord("k"): "K",
    ord("l"): "L",
    ord("m"): "M",
    ord("n"): "N",
    ord("o"): "O",
    ord("p"): "P",
    #ord("q"): "Q",
    ord("r"): "R",
    ord("s"): "S",
    ord("t"): "T",
    ord("u"): "U",
    ord("v"): "V",
    #ord("w"): "W",
    #ord("x"): "X",
    ord("y"): "Y",
    #ord("z"): "Z",
}

DIRECTOR_CURENT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(DIRECTOR_CURENT, "sign_model.pkl")
DATA_PATH = os.path.join(DIRECTOR_CURENT, "sign_data.npz")

# Incearca sa incarce mini-modelul pentru literele cu 2 maini (W, Q, X)
try:
    from Litere_cu_2_maini import (
        try_load_model as try_load_model_2maini,
        extract_hand_features as extract_hand_features_2maini,
    )
    _ml_model_2maini = try_load_model_2maini()
except ImportError:
    _ml_model_2maini = None
    extract_hand_features_2maini = None


def extract_hand_features(hand_landmarks):
    xs = []
    ys = []
    zs = []

    for lm in hand_landmarks.landmark:
        xs.append(lm.x)
        ys.append(lm.y)
        zs.append(lm.z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    xs -= xs[0]
    ys -= ys[0]
    zs -= zs[0]

    scale = max(np.max(np.abs(xs)), np.max(np.abs(ys)), np.max(np.abs(zs)), 1e-6)

    xs /= scale
    ys /= scale
    zs /= scale

    return np.concatenate([xs, ys, zs], axis=0)


def classify_static_sign_rule_based(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    extended_fingers = []

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        extended = tip.y < pip.y
        extended_fingers.append(extended)

    if all(extended_fingers):
        return "B"

    if not any(extended_fingers):
        return "A"

    return "Unknown"


def draw_hand_and_sign(image, hand_landmarks, sign_label):
    h, w, _ = image.shape

    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
    )

    wrist = hand_landmarks.landmark[0]
    x = int(wrist.x * w)
    y = int(wrist.y * h) - 10

    cv2.putText(
        image,
        f"Sign: {sign_label}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def try_load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        import joblib

        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Could not load model from {MODEL_PATH}: {e}")
        return None


def add_mirrored_hand_examples(X, y):
    X_mirrored = X.copy()
    X_mirrored[:, :21] *= -1

    X = np.vstack((X, X_mirrored))
    y = np.concatenate((y, y))

    return X, y

def train_and_save_model(X, y):
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    except ImportError:
        print(
            "scikit-learn and joblib are required to train the model. "
            "Install them with: pip install scikit-learn joblib"
        )
        return

    X, y = add_mirrored_hand_examples(X, y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")


def run_realtime_sign_detection(
    camera_index=0,
    save_images=False,
    use_ml_model=True,
    collect_mode=False,
):
    cap = cv2.VideoCapture(camera_index)

    output_dir = "Output Images"
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ml_model = try_load_model() if use_ml_model else None
    if use_ml_model and ml_model is None:
        print("No trained ML model found. Falling back to simple rule-based classifier.")

    collected_features = []
    collected_labels = []

    # In collect_mode iesirea e pe Escape ca sa poata fi folosita tasta q pentru litera Q
    if collect_mode:
        print("Collect mode ON. Tine gestul si apasa tasta literei. Escape pentru iesire.")
    else:
        print("Controls: press 'q' to quit.")

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_features_list = []

            if results.multi_hand_landmarks:
                nr_maini = len(results.multi_hand_landmarks)

                # Doua maini in modul detectie → mini-model W/Q/X
                if nr_maini == 2 and not collect_mode and _ml_model_2maini is not None and extract_hand_features_2maini is not None:
                    features_2m = extract_hand_features_2maini(results.multi_hand_landmarks)
                    pred_label = _ml_model_2maini.predict([features_2m])[0]
                    for current_hand in results.multi_hand_landmarks:
                        draw_hand_and_sign(image, current_hand, pred_label)
                else:
                    for current_hand in results.multi_hand_landmarks:
                        current_features = extract_hand_features(current_hand)
                        current_features_list.append(current_features)

                        if ml_model is not None:
                            pred_label = ml_model.predict([current_features])[0]
                        else:
                            pred_label = classify_static_sign_rule_based(current_hand)

                        draw_hand_and_sign(image, current_hand, pred_label)

            cv2.imshow("Hand / Sign Detection", image)

            key = cv2.waitKey(10) & 0xFF

            # Iesire: Escape in collect_mode, q in detection mode
            if collect_mode and key == 27:
                break
            if not collect_mode and key == ord("q"):
                break

            if collect_mode and len(current_features_list) > 0 and key in COLLECT_KEYS_TO_LABELS:
                label = COLLECT_KEYS_TO_LABELS[key]

                # Salvam doar prima mana detectata — evita dublarea datelor cand sunt 2 maini vizibile
                collected_features.append(current_features_list[0])
                collected_labels.append(label)

                print(
                    f"Am salvat un exemplu pentru litera '{label}'. Total: {len(collected_labels)}"
                )

            if save_images:
                cv2.imwrite(os.path.join(output_dir, f"{uuid.uuid1()}.jpg"), image)

    cap.release()
    cv2.destroyAllWindows()

    if collect_mode and collected_features:
        X_nou = np.vstack(collected_features)
        y_nou = np.array(collected_labels)

        if os.path.exists(DATA_PATH):
            date_vechi = np.load(DATA_PATH)

            X_vechi = date_vechi["X"]
            y_vechi = date_vechi["y"]

            X = np.vstack((X_vechi, X_nou))
            y = np.concatenate((y_vechi, y_nou))

            print(f"Am adaugat {len(y_nou)} exemple noi la cele {len(y_vechi)} vechi.")
        else:
            X = X_nou
            y = y_nou

            print(f"Am creat un fisier nou cu {len(y)} exemple.")

        np.savez(DATA_PATH, X=X, y=y)
        print(f"Salvat! Robotul cunoaste acum un total de {len(y)} exemple.")

        train_and_save_model(X, y)


if __name__ == "__main__":
    run_realtime_sign_detection(
        camera_index=0,
        save_images=False,
        use_ml_model=True,
        collect_mode=True,
    )