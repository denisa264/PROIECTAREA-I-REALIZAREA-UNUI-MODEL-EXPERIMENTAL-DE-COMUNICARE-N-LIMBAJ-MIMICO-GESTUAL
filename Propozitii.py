import os
import sys
import time
import warnings
import unicodedata
from collections import Counter, deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print("Se incarca bibliotecile pentru camera si detectie. Asteapta 15-30 secunde...", flush=True)

import cv2
import mediapipe as mp
import numpy as np

print("Bibliotecile au fost incarcate. Pornesc sistemul...", flush=True)

# Configurare

FOLDER_CURENT = os.path.dirname(os.path.abspath(__file__))

MODEL_LITERE_1M_PATH = os.path.join(FOLDER_CURENT, "sign_model.pkl")
MODEL_LITERE_2M_PATH = os.path.join(FOLDER_CURENT, "sign_model_2maini.pkl")
MODEL_EXPRESII_PATH = os.path.join(FOLDER_CURENT, "expresii_model.keras")
EXPRESII_LIST_PATH = os.path.join(FOLDER_CURENT, "expresii_list.npy")

CAMERA_INDEX = 0
WINDOW_NAME = "Propozitii - sistem automat"

LATIME_CAMERA = 640
INALTIME_CAMERA = 480

PRAG_LITERA_1M = 0.70
PRAG_LITERA_2M = 0.70
PRAG_EXPRESIE = 0.70

# MIN_CADRE_ACEEASI_LITERA = cate cadre din buffer trebuie sa coincida ca litera sa fie acceptata.
CADRE_CONFIRM_LITERA = 5
MIN_CADRE_ACEEASI_LITERA = 3
CADRE_CONFIRM_EXPRESIE = 4
MIN_CADRE_ACEEASI_EXPRESIE = 3

LEN_SECVENTA_EXPRESIE = 30
# Cat timp trebuie sa ai mainile in afara cadrului ca sistemul sa confirme automat cuvantul.
SECUNDE_FARA_MAINI_CONFIRM_CUVANT = 0.5
SECUNDE_COOLDOWN_EXPRESIE = 1.8

# Daca gesturile tale sunt mai lente, scade usor EXPRESSION_MOTION_MIN.
STATIC_MOTION_MAX = 0.012
EXPRESSION_MOTION_MIN = 0.010

# Daca este True, la fiecare ~30 cadre in care modelul de expresii face o predictie,
# se printeaza in terminal probabilitatile pentru toate clasele. Util pentru a vedea
# daca modelul tine cont de gest sau e blocat pe o singura clasa.
DEBUG_EXPRESII = True
DEBUG_INTERVAL_CADRE = 30

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

TEXT_AFISAT = {
    "BUNA": "bună",
    "MULTUMESC": "mulțumesc",
    "TE_ROG": "te rog",
    "AJUTOR": "ajutor",
    "NOROC": "noroc",
    "IUBESC": "iubesc",
    "DA": "da",
    "NU": "nu",
    "NU_STIU": "nu știu",
    "BINE": "bine",
    "RAU": "rău",
    "MERGE": "merge",
    "VREA": "vrea",
    "ARE": "are",
    "ESTE": "este",
    "CUMPARA": "cumpără",
    "MANANCA": "mănâncă",
    "BEA": "bea",
    "ACASA": "acasă",
    "SCOALA": "la școală",
    "CUMPARATURI": "la cumpărături",
    "MEDIC": "la medic",
    "APA": "apă",
    "MANCARE": "mâncare",
    "TELEFON": "telefon",
    "FOAME": "foame",
    "SETE": "sete",
    "DURERE": "durere",
    "OBOSIT": "obosit",
    "FRIG": "frig",
    "CALD": "cald",
    "MARE": "mare",
    "MIC": "mic",
    "REPEDE": "repede",
    "INCET": "încet",
    "DREAPTA": "dreapta",
    "STANGA": "stânga",
    "SUS": "sus",
    "JOS": "jos",
    "ACUM": "acum",
    "MAINE": "mâine",
    "IERI": "ieri",
    "ASTAZI": "astăzi",
    "TATA": "tată",
    "MAMA": "mamă",
    "FRATE": "frate",
    "SORA": "soră",
    "PRIETEN": "prieten",
    "UNDE": "unde",
    "CAND": "când",
    "CUM": "cum",
    "PAUZA": "",
    "STERGE": "",
    "FINAL": "",
    "CONFIRMA": "",
}

COMENZI_CONTROL = {"PAUZA", "STERGE", "FINAL", "CONFIRMA"}


def transforma_cuvant(cuvant):
    if cuvant in TEXT_AFISAT:
        return TEXT_AFISAT[cuvant]

    if cuvant.isupper() and cuvant:
        return cuvant.capitalize()

    return cuvant.lower()


def construieste_propozitie(cuvinte):
    if not cuvinte:
        return ""

    cuvinte_afisate = []
    for cuvant in cuvinte:
        text = transforma_cuvant(cuvant)
        if text:
            cuvinte_afisate.append(text)

    if not cuvinte_afisate:
        return ""

    propozitie = " ".join(cuvinte_afisate)
    return propozitie[0].upper() + propozitie[1:] + "."


def lista_expresii():
    return sorted(k for k in TEXT_AFISAT if k not in COMENZI_CONTROL)


def afiseaza_expresii_disponibile():
    expresii = [(k, v) for k, v in TEXT_AFISAT.items() if k not in COMENZI_CONTROL]
    expresii.sort()

    print("\n" + "=" * 60)
    print(f"EXPRESII DISPONIBILE ({len(expresii)} total)")
    print("=" * 60)
    for index, (cheie, valoare) in enumerate(expresii, 1):
        print(f"{index:2}. {cheie:<20} -> {valoare}")
    print("=" * 60)


# Utilitare model si text

def text_pentru_cv2(text):
    """OpenCV putText nu reda bine diacriticele; pe ecran folosim ASCII."""
    text = unicodedata.normalize("NFD", str(text))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def pune_text(imagine, text, pozitie, scala, culoare, grosime=1):
    cv2.putText(
        imagine,
        text_pentru_cv2(text),
        pozitie,
        cv2.FONT_HERSHEY_SIMPLEX,
        scala,
        culoare,
        grosime,
        cv2.LINE_AA,
    )


def pune_text_in_latime(imagine, text, x, y, latime_maxima, scala, culoare, grosime=1):
    text = text_pentru_cv2(text)
    scala_curenta = scala

    while scala_curenta > 0.38:
        latime, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scala_curenta, grosime
        )[0]
        if latime <= latime_maxima:
            break
        scala_curenta -= 0.04

    cv2.putText(
        imagine,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scala_curenta,
        culoare,
        grosime,
        cv2.LINE_AA,
    )


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
    cx, cy = w // 2, h // 2
    cv2.line(imagine, (cx - 12, cy), (cx + 12, cy), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(imagine, (cx, cy - 12), (cx, cy + 12), (80, 80, 80), 1, cv2.LINE_AA)


def incarca_model_joblib(cale, nume):
    if not os.path.exists(cale):
        print(f"[INFO] Lipseste {nume}: {cale}")
        return None

    try:
        import joblib

        model = joblib.load(cale)
        print(f"[OK] Model incarcat: {nume}")
        return model
    except Exception as eroare:
        print(f"[EROARE] Nu pot incarca {nume}: {eroare}")
        return None


def incarca_model_expresii():
    if not os.path.exists(MODEL_EXPRESII_PATH) or not os.path.exists(EXPRESII_LIST_PATH):
        print("[INFO] Modelul de expresii nu este inca disponibil.")
        print("       Dupa antrenare, pune expresii_model.keras si expresii_list.npy aici.")
        return None, None

    try:
        import importlib

        load_model = None
        for modul in ("keras.models", "tensorflow.keras.models"):
            try:
                load_model = importlib.import_module(modul).load_model
                break
            except Exception:
                continue

        if load_model is None:
            raise ImportError("Nu gasesc keras.models.load_model.")

        model = load_model(MODEL_EXPRESII_PATH)
        expresii = np.load(EXPRESII_LIST_PATH, allow_pickle=True)
        print("[OK] Model incarcat: expresii")
        return model, expresii
    except Exception as eroare:
        print(f"[EROARE] Nu pot incarca modelul de expresii: {eroare}")
        return None, None


def predictie_model_sklearn(model, vector, prag):
    if model is None:
        return "", 0.0

    try:
        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            probabilitati = model.predict_proba([vector])[0]
            index = int(np.argmax(probabilitati))
            incredere = float(probabilitati[index])
            eticheta = str(model.classes_[index]).upper()
        else:
            eticheta = str(model.predict([vector])[0]).upper()
            incredere = 1.0

        if incredere < prag:
            return "", incredere
        return eticheta, incredere
    except Exception:
        return "", 0.0


def token_stabil(buffer, minim_aparitii):
    if len(buffer) < buffer.maxlen:
        return ""

    token, aparitii = Counter(buffer).most_common(1)[0]
    if aparitii >= minim_aparitii:
        return token

    return ""

# Extragere vectori - pastrate compatibil cu scripturile de antrenare

def extrage_vector_litera(hand_landmarks):
    xs = np.array([lm.x for lm in hand_landmarks.landmark], dtype=np.float32)
    ys = np.array([lm.y for lm in hand_landmarks.landmark], dtype=np.float32)
    zs = np.array([lm.z for lm in hand_landmarks.landmark], dtype=np.float32)

    xs -= xs[0]
    ys -= ys[0]
    zs -= zs[0]

    scala = max(np.max(np.abs(xs)), np.max(np.abs(ys)), np.max(np.abs(zs)), 1e-6)

    xs /= scala
    ys /= scala
    zs /= scala

    return np.concatenate([xs, ys, zs], axis=0)


def extrage_vector_litera_2_maini(hand_landmarks_list):
    maini_sortate = sorted(hand_landmarks_list, key=lambda mana: mana.landmark[0].x)
    mana_1 = extrage_vector_litera(maini_sortate[0])
    mana_2 = extrage_vector_litera(maini_sortate[1])
    return np.concatenate([mana_1, mana_2], axis=0)


def extrage_vector_expresie(rezultate):
    # Vector redus: corp + ambele maini (fara face_landmarks).
    # Pose 0-16 include nas/ochi/urechi/gura/umeri/coate/incheieturi,
    # deci pozitia capului este pastrata fara cele 1404 valori din face mesh.
    corp = np.zeros(17 * 4, dtype=np.float32)
    if rezultate.pose_landmarks:
        puncte_corp = [
            [punct.x, punct.y, punct.z, punct.visibility]
            for index, punct in enumerate(rezultate.pose_landmarks.landmark)
            if index < 17
        ]
        if puncte_corp:
            corp = np.array(puncte_corp, dtype=np.float32).flatten()
            if len(corp) < 17 * 4:
                corp = np.pad(corp, (0, 17 * 4 - len(corp)), "constant")

    mana_stanga = (
        np.array(
            [
                [punct.x, punct.y, punct.z]
                for punct in rezultate.left_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
        if rezultate.left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    mana_dreapta = (
        np.array(
            [
                [punct.x, punct.y, punct.z]
                for punct in rezultate.right_hand_landmarks.landmark
            ],
            dtype=np.float32,
        ).flatten()
        if rezultate.right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    return np.concatenate([corp, mana_stanga, mana_dreapta])


def puncte_miscare_maini(hand_landmarks_list):
    maini_sortate = sorted(hand_landmarks_list, key=lambda mana: mana.landmark[0].x)
    puncte = []
    for mana in maini_sortate:
        puncte.extend([[lm.x, lm.y, lm.z] for lm in mana.landmark])
    return np.array(puncte, dtype=np.float32)

# Sistem integrat

class SistemPropozitii:
    def __init__(self):
        print("\n" + "=" * 70)
        print("INCARC MODELE")
        print("=" * 70)

        self.model_litere_1m = incarca_model_joblib(
            MODEL_LITERE_1M_PATH, "litere cu o mana"
        )
        self.model_litere_2m = incarca_model_joblib(
            MODEL_LITERE_2M_PATH, "litere cu doua maini Q/W/X"
        )
        self.model_expresii, self.expresii = incarca_model_expresii()

        self.propozitie = []
        self.cuvant_curent = ""

        self.buffer_litere = deque(maxlen=CADRE_CONFIRM_LITERA)
        self.buffer_expresii = deque(maxlen=CADRE_CONFIRM_EXPRESIE)
        self.secventa_expresie = deque(maxlen=LEN_SECVENTA_EXPRESIE)

        self.ultima_litera_adaugata = ""
        self.ultima_actiune = "Astept gest"
        self.detectie_curenta = ""
        self.incredere_curenta = 0.0
        self.mod_curent = "astept"

        self.maini_prezente_anterior = False
        self.ultimul_moment_cu_maini = time.time()

        self.puncte_miscare_anterioare = None
        self.istoric_miscare = deque(maxlen=8)
        self.varf_miscare = deque(maxlen=20)

        self.ultima_expresie_adaugata = ""
        self.timp_ultima_expresie = 0.0
        self.pauza_detectie_pana_la = 0.0

        # Counter pentru afisarea periodica a probabilitatilor (debug expresii).
        self.contor_debug_expresii = 0

    def proceseaza_frame_maini(self, frame, detector_maini):
        imagine_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagine_rgb = cv2.flip(imagine_rgb, 1)

        imagine_rgb.flags.writeable = False
        rezultate = detector_maini.process(imagine_rgb)
        imagine_rgb.flags.writeable = True

        imagine_bgr = cv2.cvtColor(imagine_rgb, cv2.COLOR_RGB2BGR)
        return imagine_bgr, rezultate

    def proceseaza_frame_expresii(self, frame, detector_holistic):
        imagine_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        imagine_rgb.flags.writeable = False
        rezultate = detector_holistic.process(imagine_rgb)
        imagine_rgb.flags.writeable = True

        return rezultate

    def calculeaza_miscare(self, hand_landmarks_list):
        puncte_curente = puncte_miscare_maini(hand_landmarks_list)

        if (
            self.puncte_miscare_anterioare is None
            or self.puncte_miscare_anterioare.shape != puncte_curente.shape
        ):
            self.puncte_miscare_anterioare = puncte_curente
            return 0.0, 0.0

        diferente = puncte_curente - self.puncte_miscare_anterioare
        miscare_frame = float(np.mean(np.linalg.norm(diferente, axis=1)))
        self.puncte_miscare_anterioare = puncte_curente

        self.istoric_miscare.append(miscare_frame)
        miscare_medie = float(np.mean(self.istoric_miscare)) if self.istoric_miscare else 0.0

        self.varf_miscare.append(miscare_medie)
        miscare_varf = float(max(self.varf_miscare)) if self.varf_miscare else 0.0

        return miscare_medie, miscare_varf

    def actualizeaza_litera(self, litera, incredere):
        if not litera:
            return

        self.buffer_litere.append(litera)
        litera_stabila = token_stabil(self.buffer_litere, MIN_CADRE_ACEEASI_LITERA)

        if not litera_stabila:
            return

        if litera_stabila != self.ultima_litera_adaugata:
            self.cuvant_curent += litera_stabila
            self.ultima_litera_adaugata = litera_stabila
            self.ultima_actiune = f"Litera adaugata: {litera_stabila}"
            print(f"[LITERA] {litera_stabila} -> {self.cuvant_curent}")

        self.detectie_curenta = litera_stabila
        self.incredere_curenta = incredere
        self.buffer_litere.clear()

    def confirma_cuvant_curent(self):
        if not self.cuvant_curent:
            return False

        self.propozitie.append(self.cuvant_curent)
        self.ultima_actiune = f"Cuvant confirmat: {self.cuvant_curent}"
        print(f"[CUVANT] {self.cuvant_curent}")
        print(f"[PROPOZITIE] {construieste_propozitie(self.propozitie)}")

        self.cuvant_curent = ""
        self.ultima_litera_adaugata = ""
        self.buffer_litere.clear()
        return True

    def sterge_ultimul_element(self):
        if self.cuvant_curent:
            sters = self.cuvant_curent[-1]
            self.cuvant_curent = self.cuvant_curent[:-1]
            self.ultima_litera_adaugata = self.cuvant_curent[-1] if self.cuvant_curent else ""
            self.ultima_actiune = f"Litera stearsa: {sters}"
            print(f"[STERGE] litera {sters}")
            return

        if self.propozitie:
            sters = self.propozitie.pop()
            self.ultima_actiune = f"Token sters: {sters}"
            print(f"[STERGE] {sters}")
            print(f"[PROPOZITIE] {construieste_propozitie(self.propozitie)}")
            return

        self.ultima_actiune = "Nu am ce sterge"

    def trateaza_expresie_confirmata(self, expresie):
        acum = time.time()
        if acum - self.timp_ultima_expresie < SECUNDE_COOLDOWN_EXPRESIE:
            return

        self.timp_ultima_expresie = acum
        self.ultima_expresie_adaugata = expresie

        if expresie == "STERGE":
            self.sterge_ultimul_element()
        elif expresie == "PAUZA":
            self.pauza_detectie_pana_la = acum + 2.0
            self.ultima_actiune = "Pauza scurta"
        elif expresie in ("FINAL", "CONFIRMA"):
            self.confirma_cuvant_curent()
            text_final = construieste_propozitie(self.propozitie)
            self.ultima_actiune = f"Final: {text_final}" if text_final else "Final gol"
            print(f"[FINAL] {text_final}")
        else:
            self.confirma_cuvant_curent()
            self.propozitie.append(expresie)
            self.ultima_actiune = f"Expresie adaugata: {expresie}"
            print(f"[EXPRESIE] {expresie}")
            print(f"[PROPOZITIE] {construieste_propozitie(self.propozitie)}")

        self.buffer_expresii.clear()
        self.secventa_expresie.clear()
        self.varf_miscare.clear()

    def actualizeaza_expresie(self, rezultate_holistic, miscare_varf):
        if self.model_expresii is None or self.expresii is None:
            self.ultima_actiune = "Model expresii lipsa"
            return False

        vector = extrage_vector_expresie(rezultate_holistic)
        self.secventa_expresie.append(vector)

        if len(self.secventa_expresie) < LEN_SECVENTA_EXPRESIE:
            return False

        secventa = np.array(list(self.secventa_expresie), dtype=np.float32)
        predictie = self.model_expresii.predict(np.expand_dims(secventa, axis=0), verbose=0)[0]
        index = int(np.argmax(predictie))
        incredere = float(predictie[index])
        expresie = str(self.expresii[index]).upper()

        # --- Debug: arata probabilitatile pentru toate clasele, periodic ---
        if DEBUG_EXPRESII:
            self.contor_debug_expresii += 1
            if self.contor_debug_expresii >= DEBUG_INTERVAL_CADRE:
                self.contor_debug_expresii = 0
                perechi = sorted(
                    zip(self.expresii, predictie),
                    key=lambda pereche: pereche[1],
                    reverse=True,
                )
                detalii = "  ".join(
                    f"{str(eticheta).upper()}={prob * 100:5.1f}%"
                    for eticheta, prob in perechi
                )
                print(f"[DEBUG] miscare_varf={miscare_varf:.4f}  |  {detalii}")

        self.detectie_curenta = expresie
        self.incredere_curenta = incredere

        if incredere < PRAG_EXPRESIE or miscare_varf < EXPRESSION_MOTION_MIN:
            return False

        self.buffer_expresii.append(expresie)
        expresie_stabila = token_stabil(self.buffer_expresii, MIN_CADRE_ACEEASI_EXPRESIE)

        if expresie_stabila:
            self.trateaza_expresie_confirmata(expresie_stabila)
            return True

        return False

    def deseneaza_maini(self, imagine, hand_landmarks_list):
        for mana in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                imagine,
                mana,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2),
            )

    def proceseaza_detectie(self, frame, detector_maini, detector_holistic):
        imagine, rezultate_maini = self.proceseaza_frame_maini(frame, detector_maini)

        hand_landmarks_list = (
            list(rezultate_maini.multi_hand_landmarks)
            if rezultate_maini.multi_hand_landmarks
            else []
        )
        numar_maini = len(hand_landmarks_list)
        acum = time.time()

        self.detectie_curenta = ""
        self.incredere_curenta = 0.0

        if numar_maini:
            self.maini_prezente_anterior = True
            self.ultimul_moment_cu_maini = acum
            self.deseneaza_maini(imagine, hand_landmarks_list)
        else:
            self.puncte_miscare_anterioare = None
            self.istoric_miscare.clear()
            self.varf_miscare.clear()

            if (
                self.maini_prezente_anterior
                and acum - self.ultimul_moment_cu_maini >= SECUNDE_FARA_MAINI_CONFIRM_CUVANT
            ):
                self.confirma_cuvant_curent()
                self.maini_prezente_anterior = False
                self.secventa_expresie.clear()
                self.buffer_expresii.clear()
                self.buffer_litere.clear()
                self.ultima_litera_adaugata = ""

            self.mod_curent = "astept"
            self.deseneaza_hud(imagine)
            return imagine

        if acum < self.pauza_detectie_pana_la:
            self.mod_curent = "pauza"
            self.deseneaza_hud(imagine)
            return imagine

        miscare_medie, miscare_varf = self.calculeaza_miscare(hand_landmarks_list)
        expresie_confirmata = False

        # Modelul de expresii ruleaza permanent in paralel cu cel de litere.
        # Cand este detectata o expresie reala (incredere mare + miscare ampla),cuvantul curent este confirmat automat in actualizeaza_expresie, asa ca semnatarul nu mai trebuie sa scoata mainile din cadru.
        if self.model_expresii is not None and self.expresii is not None:
            rezultate_holistic = self.proceseaza_frame_expresii(
                frame, detector_holistic
            )
            expresie_confirmata = self.actualizeaza_expresie(
                rezultate_holistic, miscare_varf
            )

        if expresie_confirmata:
            self.mod_curent = "expresie"
            self.buffer_litere.clear()
            self.ultima_litera_adaugata = ""

        elif numar_maini == 1:
            if miscare_medie <= STATIC_MOTION_MAX:
                self.mod_curent = "litere 1 mana"
                vector = extrage_vector_litera(hand_landmarks_list[0])
                litera, incredere = predictie_model_sklearn(
                    self.model_litere_1m, vector, PRAG_LITERA_1M
                )
                self.detectie_curenta = litera or "?"
                self.incredere_curenta = incredere
                self.actualizeaza_litera(litera, incredere)
            else:
                self.mod_curent = "miscare"
                self.buffer_litere.clear()
                self.ultima_litera_adaugata = ""

        elif numar_maini == 2:
            if miscare_medie <= STATIC_MOTION_MAX:
                self.mod_curent = "litere 2 maini"
                vector = extrage_vector_litera_2_maini(hand_landmarks_list)
                litera, incredere = predictie_model_sklearn(
                    self.model_litere_2m, vector, PRAG_LITERA_2M
                )
                self.detectie_curenta = litera or "?"
                self.incredere_curenta = incredere
                self.actualizeaza_litera(litera, incredere)
            else:
                self.mod_curent = "miscare"
                self.buffer_litere.clear()
                self.ultima_litera_adaugata = ""

        self.afiseaza_detectie_pe_maini(imagine, hand_landmarks_list, miscare_medie)

        self.deseneaza_hud(imagine)
        return imagine

    def afiseaza_detectie_pe_maini(self, imagine, hand_landmarks_list, miscare):
        if not self.detectie_curenta:
            return

        h, w, _ = imagine.shape
        text = f"{self.detectie_curenta} {self.incredere_curenta * 100:.0f}%"
        if self.mod_curent == "expresie":
            text += f" | miscare {miscare:.3f}"

        for mana in hand_landmarks_list:
            wrist = mana.landmark[0]
            x = max(10, min(int(wrist.x * w), w - 250))
            y = max(120, int(wrist.y * h) - 15)
            pune_text(imagine, text, (x, y), 0.65, (0, 255, 0), 2)

    def deseneaza_hud(self, imagine):
        h, w, _ = imagine.shape
        cv2.rectangle(imagine, (0, 0), (w, 64), (20, 20, 20), -1)
        cv2.rectangle(imagine, (0, h - 48), (w, h), (20, 20, 20), -1)

        propozitie = construieste_propozitie(self.propozitie)
        cuvant = self.cuvant_curent + "_" if self.cuvant_curent else "_"

        pune_text_in_latime(
            imagine,
            propozitie,
            14,
            42,
            w - 28,
            0.92,
            (255, 255, 255),
            2,
        )
        pune_text_in_latime(
            imagine,
            f"Cuvant: {cuvant}",
            14,
            h - 16,
            w - 28,
            0.62,
            (120, 230, 120),
            2,
        )

        # Indicator de stare in coltul dreapta-sus. Modelul de expresii ruleaza permanent in paralel; il avertizam pe utilizator daca un cuvant in curs
        # va fi confirmat automat la urmatoarea expresie detectata.
        if self.cuvant_curent:
            text_mod = f"MOD: LITERE  ({self.cuvant_curent} se confirma la expresie)"
            culoare_mod = (120, 200, 255)  # albastru deschis
        else:
            text_mod = "MOD: LITERE + EXPRESII (paralel)"
            culoare_mod = (140, 220, 140)  # verde

        # Calculam latimea textului ca sa il aliniem la dreapta
        (text_w, _), _ = cv2.getTextSize(
            text_mod, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        pune_text(
            imagine,
            text_mod,
            (w - text_w - 14, 24),
            0.55,
            culoare_mod,
            1,
        )

    def afiseaza_ecran_final(self, text_final):
        latime, inaltime = 980, 260
        canvas = np.zeros((inaltime, latime, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)

        pune_text(canvas, "REZULTAT FINAL", (28, 54), 0.9, (140, 220, 140), 2)
        pune_text_in_latime(canvas, text_final or "[gol]", 28, 132, latime - 56, 1.15, (255, 255, 255), 2)
        pune_text(canvas, "Se inchide automat...", (28, 215), 0.55, (170, 170, 170), 1)

        cv2.imshow("Rezultat final", canvas)
        cv2.waitKey(2500)

    def ruleaza(self):
        # Pe Windows folosim DSHOW pentru ca initializeaza camera mult mai rapid
        # si accepta mai consistent rezolutiile HD.
        if os.name == "nt":
            camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            print("[EROARE] Nu pot porni camera.")
            return

        # Cerem rezolutia HD. Daca webcamul nu o accepta, ramane la cat poate.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, LATIME_CAMERA)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, INALTIME_CAMERA)
        latime_efectiva = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        inaltime_efectiva = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera ruleaza la {latime_efectiva}x{inaltime_efectiva}.")

        # Fereastra redimensionabila pornita la rezolutia camerei.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, LATIME_CAMERA, INALTIME_CAMERA)

        print("\n" + "=" * 70)
        print("SISTEM PORNIT")
        print("=" * 70)
        print("Ridica mana pentru litere.")
        print("Coboara/ascunde mana 1-2 secunde pentru confirmarea cuvantului.")
        print("Pentru Q/W/X foloseste doua maini tinute relativ fix.")
        print("Pentru expresii foloseste doua maini cu miscare mai ampla.")
        print("Pentru corectare, antreneaza/foloseste expresia STERGE.")
        print("ESC inchide programul si afiseaza propozitia finala.")
        print("=" * 70 + "\n")

        with mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        ) as detector_maini, mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as detector_holistic:
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break

                imagine = self.proceseaza_detectie(frame, detector_maini, detector_holistic)
                deseneaza_ghidaj_incadrare(imagine)
                cv2.imshow(WINDOW_NAME, imagine)

                tasta = cv2.waitKey(10) & 0xFF
                if tasta == 27:
                    break

                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

        self.confirma_cuvant_curent()
        text_final = construieste_propozitie(self.propozitie)

        print("\n" + "=" * 70)
        print("REZULTAT FINAL")
        print("=" * 70)
        print(text_final)
        print("=" * 70 + "\n")

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sistem = SistemPropozitii()
    sistem.ruleaza()
