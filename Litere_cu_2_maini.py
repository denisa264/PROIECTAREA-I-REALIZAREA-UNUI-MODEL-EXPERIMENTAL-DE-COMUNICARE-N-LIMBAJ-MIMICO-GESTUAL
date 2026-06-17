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
    ord("q"): "Q",
    ord("w"): "W",
    ord("x"): "X",
}

DIRECTOR_CURENT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(DIRECTOR_CURENT, "sign_model_2maini.pkl")
DATA_PATH = os.path.join(DIRECTOR_CURENT, "sign_data_2maini.npz")


def extract_single_hand_features(hand_landmarks):
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


def extract_hand_features(hand_landmarks_list):
    # Sorteaza mainile dupa pozitia pe axa X (stanga → dreapta pe ecran)
    # pentru a avea mereu acelasi vector indiferent de ordinea detectiei
    sorted_hands = sorted(hand_landmarks_list, key=lambda lm: lm.landmark[0].x)
    v1 = extract_single_hand_features(sorted_hands[0])
    v2 = extract_single_hand_features(sorted_hands[1])
    return np.concatenate([v1, v2])


def classify_static_sign_rule_based(hand_landmarks):
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
    # Vectorul are 126 valori: [mana1_xs(21), mana1_ys(21), mana1_zs(21), mana2_xs(21), mana2_ys(21), mana2_zs(21)]
    # La oglindire inversam X-urile ambelor maini si schimbam ordinea lor
    X_mirrored = X.copy()
    X_mirrored[:, :21] *= -1       # X-urile mainii 1
    X_mirrored[:, 63:84] *= -1     # X-urile mainii 2

    # Schimbam ordinea mainilor (stanga devine dreapta dupa oglindire)
    X_swapped = np.hstack([X_mirrored[:, 63:], X_mirrored[:, :63]])

    X = np.vstack((X, X_swapped))
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

    # Iesirea e mereu pe Escape — q si w sunt taste pentru litere Q si W
    print("Controls: press Escape to quit.")

    if collect_mode:
        print("Collect mode ON. Arata AMBELE maini, tine gestul si apasa q / w / x.")

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

            current_features = None
            nr_maini = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            if nr_maini == 2:
                current_features = extract_hand_features(results.multi_hand_landmarks)

                if ml_model is not None:
                    pred_label = ml_model.predict([current_features])[0]
                else:
                    pred_label = classify_static_sign_rule_based(results.multi_hand_landmarks[0])

                for current_hand in results.multi_hand_landmarks:
                    draw_hand_and_sign(image, current_hand, pred_label)

            elif nr_maini == 1:
                # O singura mana vizibila — arata si a doua mana
                draw_hand_and_sign(image, results.multi_hand_landmarks[0], "Arata ambele maini")

            cv2.imshow("Hand / Sign Detection - 2 Maini", image)

            key = cv2.waitKey(10) & 0xFF

            if key == 27:  # Escape
                break

            if collect_mode and current_features is not None and key in COLLECT_KEYS_TO_LABELS:
                label = COLLECT_KEYS_TO_LABELS[key]

                collected_features.append(current_features)
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
        print(f"Salvat! Modelul cunoaste acum un total de {len(y)} exemple.")

        train_and_save_model(X, y)


if __name__ == "__main__":
    run_realtime_sign_detection(
        camera_index=0,
        save_images=False,
        use_ml_model=True,
        collect_mode=True,
    )
