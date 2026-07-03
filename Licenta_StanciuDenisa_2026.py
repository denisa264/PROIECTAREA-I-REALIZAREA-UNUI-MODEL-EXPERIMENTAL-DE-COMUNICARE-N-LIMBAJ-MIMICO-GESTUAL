from flask import Flask, render_template, request, Response, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import os
import time
from datetime import datetime
import cv2
import mediapipe as mp

from Propozitii import (
    SistemPropozitii,
    deseneaza_ghidaj_incadrare,
    construieste_propozitie,
    LATIME_CAMERA,
    INALTIME_CAMERA,
)

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

app = Flask(__name__)

app.config['SECRET_KEY'] = 'cheie_secreta_licenta_stanciu_denisa_2026'

DIRECTOR_PROIECT = os.path.dirname(os.path.abspath(__file__))
CALE_BAZA_DATE = os.path.join(DIRECTOR_PROIECT, 'licenta.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{CALE_BAZA_DATE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Utilizator(db.Model):
    uuid = db.Column(db.String(16), primary_key=True)
    nume = db.Column(db.String(50), nullable=False)
    prenume = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    telefon = db.Column(db.String(20), unique=True, nullable=False)
    parola = db.Column(db.String(200), nullable=False)

    sesiuni = db.relationship(
        'Sesiune',
        backref='utilizator',
        lazy=True,
        cascade='all, delete-orphan'
    )

class Sesiune(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    utilizator_uuid = db.Column(
        db.String(16),
        db.ForeignKey('utilizator.uuid'),
        nullable=False
    )
    data_start = db.Column(db.DateTime, default=datetime.now, nullable=False)
    data_end = db.Column(db.DateTime, nullable=True)

    mesaje = db.relationship(
        'Mesaj',
        backref='sesiune',
        lazy=True,
        cascade='all, delete-orphan',
        order_by='Mesaj.data'
    )

class Mesaj(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sesiune_id = db.Column(
        db.Integer,
        db.ForeignKey('sesiune.id'),
        nullable=False
    )
    # 'microfon' sau 'semne', spune cine a vorbit.
    sursa = db.Column(db.String(20), nullable=False)
    text = db.Column(db.String(500), nullable=False)
    data = db.Column(db.DateTime, default=datetime.now, nullable=False)


with app.app_context():
    db.create_all()

sistem_recunoastere = None
camera_curenta = None


def obtine_sistem_recunoastere():
    """Lazy load: incarc modelele abia cand utilizatorul cere prima data camera."""
    global sistem_recunoastere
    if sistem_recunoastere is None:
        sistem_recunoastere = SistemPropozitii()
    return sistem_recunoastere


def reseteaza_stare_recunoastere():
    """Sterge propozitia curenta si bufferele, pregatind o sesiune noua."""
    if sistem_recunoastere is None:
        return
    sistem_recunoastere.propozitie = []
    sistem_recunoastere.cuvant_curent = ""
    sistem_recunoastere.ultima_litera_adaugata = ""
    sistem_recunoastere.buffer_litere.clear()
    sistem_recunoastere.buffer_expresii.clear()
    sistem_recunoastere.secventa_expresie.clear()
    sistem_recunoastere.maini_prezente_anterior = False


def salveaza_propozitie_detectata():
    """Confirma cuvantul curent, construieste propozitia finala si o stocheaza
    in baza de date ca mesaj cu sursa 'semne'. Intoarce textul salvat sau None."""
    if sistem_recunoastere is None:
        return None

    sistem_recunoastere.confirma_cuvant_curent()
    text = construieste_propozitie(sistem_recunoastere.propozitie)
    if not text:
        return None

    utilizator = utilizator_conectat()
    if utilizator is None:
        return None

    sesiune = sesiune_activa(utilizator)
    mesaj_nou = Mesaj(
        sesiune_id=sesiune.id,
        sursa='semne',
        text=text[:500]
    )
    db.session.add(mesaj_nou)
    db.session.commit()
    return text


def genereaza_cadre():
    """Generatorul care produce cadre JPEG pentru browser, cu recunoastere live."""
    global camera_curenta

    sistem = obtine_sistem_recunoastere()

    if os.name == 'nt':
        camera_curenta = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        camera_curenta = cv2.VideoCapture(0)

    if not camera_curenta.isOpened():
        camera_curenta = None
        return

    camera_curenta.set(cv2.CAP_PROP_FRAME_WIDTH, LATIME_CAMERA)
    camera_curenta.set(cv2.CAP_PROP_FRAME_HEIGHT, INALTIME_CAMERA)

    FPS_TINTA = 10
    timp_intre_cadre = 1.0 / FPS_TINTA

    with mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
        max_num_hands=2,
        model_complexity=1,
    ) as detector_maini, mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
        refine_face_landmarks=False,
    ) as detector_holistic:
        timp_ultim_cadru = 0.0
        while camera_curenta is not None and camera_curenta.isOpened():
            ok, frame = camera_curenta.read()
            if not ok:
                break

            acum = time.time()
            de_asteptat = timp_intre_cadre - (acum - timp_ultim_cadru)
            if de_asteptat > 0:
                time.sleep(de_asteptat)
            timp_ultim_cadru = time.time()

            imagine = sistem.proceseaza_detectie(
                frame, detector_maini, detector_holistic
            )
            deseneaza_ghidaj_incadrare(imagine)

            ok_jpg, buffer = cv2.imencode('.jpg', imagine)
            if not ok_jpg:
                continue

            yield (
                b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                + buffer.tobytes()
                + b'\r\n'
            )


@app.route('/video_feed')
def video_feed():
    return Response(
        genereaza_cadre(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/opreste_camera', methods=['POST'])
def opreste_camera():
    """Apelat din browser la apasarea butonului Pauza camera.
    Salveaza propozitia detectata in baza de date, elibereaza camera si
    reseteaza starea sistemului pentru o sesiune noua."""
    global camera_curenta

    text_salvat = salveaza_propozitie_detectata()

    if camera_curenta is not None:
        camera_curenta.release()
    camera_curenta = None

    reseteaza_stare_recunoastere()

    return jsonify({'succes': True, 'text_salvat': text_salvat})


@app.route('/text_semne_curent')
def text_semne_curent():
    """Returneaza propozitia construita in acest moment din semne, fara sa o
    salveze in baza de date. Folosit de browser pentru text-to-speech in timpul
    sesiunii. Spre deosebire de opreste_camera, nu confirma cuvantul curent ca
    sa nu intervina in fluxul de detectare."""
    if sistem_recunoastere is None:
        return jsonify({'text': '', 'cuvant_in_curs': ''})

    text = construieste_propozitie(sistem_recunoastere.propozitie)
    cuvant_in_curs = sistem_recunoastere.cuvant_curent or ''

    return jsonify({'text': text, 'cuvant_in_curs': cuvant_in_curs})

def utilizator_conectat():
    """Intoarce utilizatorul logat curent sau None daca nimeni nu e conectat."""
    uuid_curent = session.get('utilizator_uuid')
    if not uuid_curent:
        return None
    return Utilizator.query.get(uuid_curent)


def sesiune_activa(utilizator):
    """Returneaza sesiunea activa (fara data_end) a utilizatorului.
    Daca nu exista, creeaza una noua si o intoarce."""
    sesiune = Sesiune.query.filter_by(
        utilizator_uuid=utilizator.uuid,
        data_end=None
    ).order_by(Sesiune.data_start.desc()).first()

    if sesiune is None:
        sesiune = Sesiune(utilizator_uuid=utilizator.uuid)
        db.session.add(sesiune)
        db.session.commit()

    return sesiune


@app.route('/')
def pagina_principala():

    if utilizator_conectat():
        return redirect(url_for('aplicatie'))
    return render_template('pagina_principala.html')


@app.route('/termeni')
def termeni():
    return render_template('termeni.html')


@app.route('/aplicatie')
def aplicatie():
    utilizator = utilizator_conectat()
    if not utilizator:
        return redirect(url_for('pagina_principala'))

    sesiune = sesiune_activa(utilizator)
    return render_template(
        'interfata_preluare_date.html',
        utilizator=utilizator,
        sesiune=sesiune
    )


@app.route('/login', methods=['POST'])
def login():
    date_introdus = request.form.get('cont')
    parola_introdusa = request.form.get('parola')

    if not date_introdus or not parola_introdusa:
        return render_template(
            'pagina_principala.html',
            mesaj_rosu="Campurile sunt obligatorii!"
        )

    utilizator_gasit = Utilizator.query.filter(
        (Utilizator.email == date_introdus) | (Utilizator.telefon == date_introdus)
    ).first()

    if utilizator_gasit and check_password_hash(utilizator_gasit.parola, parola_introdusa):
        # Salvam uuid ul in sesiunea Flask. Ramane activ pe browser pana la logout.
        session['utilizator_uuid'] = utilizator_gasit.uuid
        return redirect(url_for('aplicatie'))
    else:
        return render_template(
            'pagina_principala.html',
            mesaj_rosu="Date incorecte!"
        )


@app.route('/inregistrare', methods=['POST'])
def inregistrare():
    nume = request.form.get('nume')
    prenume = request.form.get('prenume')
    email = request.form.get('email')
    telefon = request.form.get('telefon')
    parola = request.form.get('parola')

    if not nume or not prenume or not email or not telefon or not parola:
        return render_template(
            'pagina_principala.html',
            mesaj_rosu="Completeaza toate campurile!"
        )

    parola_criptata = generate_password_hash(parola)
    cod_generat = str(uuid.uuid4().hex)[:16]

    try:
        cont_nou = Utilizator(
            uuid=cod_generat,
            nume=nume,
            prenume=prenume,
            email=email,
            telefon=telefon,
            parola=parola_criptata
        )
        db.session.add(cont_nou)
        db.session.commit()

        session['utilizator_uuid'] = cont_nou.uuid
        return redirect(url_for('aplicatie'))
    except Exception:
        db.session.rollback()
        return render_template(
            'pagina_principala.html',
            mesaj_rosu="Email sau telefon deja existent!"
        )


@app.route('/logout', methods=['POST'])
def logout():

    global camera_curenta
    salveaza_propozitie_detectata()
    if camera_curenta is not None:
        camera_curenta.release()
    camera_curenta = None
    reseteaza_stare_recunoastere()

    session.pop('utilizator_uuid', None)
    return redirect(url_for('pagina_principala'))

@app.route('/salveaza_mesaj', methods=['POST'])
def salveaza_mesaj():
    utilizator = utilizator_conectat()
    if not utilizator:
        return jsonify({'succes': False, 'eroare': 'Nu esti conectat.'}), 401

    text = (request.form.get('text') or '').strip()
    sursa = request.form.get('sursa', 'microfon')

    if not text:
        return jsonify({'succes': False, 'eroare': 'Mesaj gol.'}), 400

    if sursa not in ('microfon', 'semne'):
        sursa = 'microfon'

    sesiune = sesiune_activa(utilizator)

    mesaj_nou = Mesaj(
        sesiune_id=sesiune.id,
        sursa=sursa,
        text=text[:500]
    )
    db.session.add(mesaj_nou)
    db.session.commit()

    return jsonify({
        'succes': True,
        'id': mesaj_nou.id,
        'sursa': mesaj_nou.sursa,
        'text': mesaj_nou.text,
        'data': mesaj_nou.data.strftime('%H:%M:%S')
    })


@app.route('/mesaje_sesiune_curenta')
def mesaje_sesiune_curenta():
    """Returneaza mesajele din sesiunea activa, in format JSON.
    Folosit de pagina aplicatiei pentru a afisa conversatia in timp real."""
    utilizator = utilizator_conectat()
    if not utilizator:
        return jsonify({'mesaje': []}), 401

    sesiune = sesiune_activa(utilizator)
    lista = [
        {
            'id': m.id,
            'sursa': m.sursa,
            'text': m.text,
            'data': m.data.strftime('%H:%M:%S')
        }
        for m in sesiune.mesaje
    ]
    return jsonify({'mesaje': lista, 'sesiune_id': sesiune.id})


@app.route('/incheie_sesiune', methods=['POST'])
def incheie_sesiune():
    utilizator = utilizator_conectat()
    if not utilizator:
        return redirect(url_for('pagina_principala'))

    global camera_curenta
    salveaza_propozitie_detectata()
    if camera_curenta is not None:
        camera_curenta.release()
    camera_curenta = None
    reseteaza_stare_recunoastere()

    sesiune = Sesiune.query.filter_by(
        utilizator_uuid=utilizator.uuid,
        data_end=None
    ).order_by(Sesiune.data_start.desc()).first()

    if sesiune is not None:
        sesiune.data_end = datetime.now()
        db.session.commit()

    return redirect(url_for('istoric'))

@app.route('/istoric')
def istoric():
    utilizator = utilizator_conectat()
    if not utilizator:
        return redirect(url_for('pagina_principala'))

    # Aducem toate sesiunile in ordine descrescatoare (cele recente primele).
    sesiuni = Sesiune.query.filter_by(
        utilizator_uuid=utilizator.uuid
    ).order_by(Sesiune.data_start.desc()).all()

    return render_template('istoric.html', utilizator=utilizator, sesiuni=sesiuni)

@app.route('/stergere_cont', methods=['GET', 'POST'])
def stergere_cont():
    if request.method == 'GET':
        return render_template('stergere_cont.html')

    date_introdus = request.form.get('cont')
    parola_introdusa = request.form.get('parola')

    utilizator_gasit = Utilizator.query.filter(
        (Utilizator.email == date_introdus) | (Utilizator.telefon == date_introdus)
    ).first()

    if utilizator_gasit and check_password_hash(utilizator_gasit.parola, parola_introdusa):
 
        db.session.delete(utilizator_gasit)
        db.session.commit()
        session.pop('utilizator_uuid', None)
        mesaj = "Contul si toate datele tale au fost sterse cu succes din sistem."
        return render_template('pagina_principala.html', mesaj_verde=mesaj)
    else:
        eroare = "Datele introduse sunt incorecte! Nu am putut confirma identitatea pentru stergere."
        return render_template('stergere_cont.html', mesaj_rosu=eroare)


if __name__ == '__main__':
    app.run(debug=True)
