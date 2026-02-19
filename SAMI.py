import cv2
import numpy as np
import pickle
import os
import json
from datetime import datetime
import face_recognition

# FUNCIÓN DE CONVERSIÓN

db_pkl = '/home/pi/Documents/andyresi/personas_db.pkl'
destino = '/home/pi/Documents/andyresi/SAMI/backend/data/reporte.json'

def generar_reporte_json(archivo_entrada=db_pkl, archivo_salida=destino):
    if not os.path.exists(archivo_entrada):
        print(f"Error: No se encuentra {archivo_entrada}")
        return

    with open(archivo_entrada, 'rb') as f:
        data = pickle.load(f)

    nombres = data.get('names', [])
    metadatos = data.get('metadata', [])

    # Inicializamos contadores
    total_hombres = 0
    total_mujeres = 0

    # Iteramos sobre los metadatos para contar géneros
    for registro in metadatos:
        prediccion = registro.get('prediction', "").lower()
        if "hombre" in prediccion:
            total_hombres += 1
        elif "mujer" in prediccion:
            total_mujeres += 1

    reporte = {
        "total_participantes": len(nombres),
        "hombres": total_hombres,
        "mujeres": total_mujeres
    }

    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=4, ensure_ascii=False)
    print(f"✅ Reporte generado: {archivo_salida} (H: {total_hombres}, M: {total_mujeres})")

# LÓGICA DE PREDICCIÓN Y MODELOS

AGE_LIST = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
GENDER_LIST = ['Hombre', 'Mujer']

def load_ai_models():
    models_dir = '/home/pi/Documents/andyresi/models'
    age_proto = os.path.join(models_dir, 'age_deploy.prototxt')
    age_model = os.path.join(models_dir, 'age_net.caffemodel')
    gender_proto = os.path.join(models_dir, 'gender_deploy.prototxt')
    gender_model = os.path.join(models_dir, 'gender_net.caffemodel')

    try:
        age_net = cv2.dnn.readNet(age_proto, age_model)
        gender_net = cv2.dnn.readNet(gender_proto, gender_model)
        return age_net, gender_net
    except Exception as e:
        print(f"Error cargando modelos de IA: {e}")
        return None, None

def predict_age_gender(face_img, age_net, gender_net):
    try:
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                    (78.4263377603, 87.7689143744, 114.895847746),
                                    swapRB=False)
        gender_net.setInput(blob)
        gender_idx = gender_net.forward()[0].argmax()
        age_net.setInput(blob)
        age_idx = age_net.forward()[0].argmax()
        return f"{GENDER_LIST[gender_idx]}, {AGE_LIST[age_idx]}"
    except:
        return "Error"

# GESTIÓN DE BASE DE DATOS

class PersonDatabase:
    def __init__(self, db_file='/home/pi/Documents/andyresi/personas_db.pkl'):
        self.db_file = db_file
        self.known_faces, self.known_names, self.known_metadata = [], [], []
        self.next_id = 1
        if os.path.exists(db_file):
            with open(db_file, 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data['faces']
                self.known_names = data['names']
                self.known_metadata = data['metadata']
                self.next_id = data.get('next_id', len(self.known_names) + 1)
    
    def save_database(self):
        with open(self.db_file, 'wb') as f:
            pickle.dump({'faces': self.known_faces, 'names': self.known_names, 
                         'metadata': self.known_metadata, 'next_id': self.next_id}, f)
    
    def add_person(self, face_encoding, prediction):
        name = f"Persona_{self.next_id}"
        self.next_id += 1
        self.known_faces.append(face_encoding)
        self.known_names.append(name)
        self.known_metadata.append({
            'first_seen': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_visits': 1,
            'prediction': prediction
        })
        self.save_database()
        return name

# CONTROL DE INTERFAZ (BOTÓN)

boton_presionado = False
def manejar_clic(event, x, y, flags, param):
    global boton_presionado

    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 140 and 10 <= y <= 50:
            boton_presionado = True

def main():
    global boton_presionado
    db = PersonDatabase()
    age_net, gender_net = load_ai_models()
    if not age_net: return

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('SAMI')
    cv2.setMouseCallback('SAMI', manejar_clic)
    
    current_frame_faces = {}
    frame_count = 0

    print("Sistema Iniciado. Haz clic en 'TERMINAR' para cerrar y guardar")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or boton_presionado: break
        
        frame_count += 1
        
        if frame_count % 3 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            current_frame_faces = {}
            for encoding, location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(db.known_faces, encoding, tolerance=0.6)
                if True in matches:
                    idx = matches.index(True)
                    name = db.known_names[idx]
                    db.known_metadata[idx]['total_visits'] += 1
                    info = db.known_metadata[idx].get('prediction', "Registrado")
                else:
                    top, right, bottom, left = [c * 4 for c in location]
                    info = predict_age_gender(frame[top:bottom, left:right], age_net, gender_net)
                    name = db.add_person(encoding, info)
                current_frame_faces[name] = {'box': [c * 4 for c in location], 'info': info}

        # DIBUJAR BOTÓN "TERMINAR"
        cv2.rectangle(frame, (10, 10), (140, 50), (0, 0, 200), -1)
        cv2.putText(frame, "TERMINAR", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Dibujar Rostros
        for name, data in current_frame_faces.items():
            top, right, bottom, left = data['box']
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {data['info']}", (left, top-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('SAMI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # FINALIZACIÓN
    cap.release()
    cv2.destroyAllWindows()
    
    if boton_presionado:
        print("Botón presionado. Convirtiendo base de datos...")
        generar_reporte_json()

if __name__ == "__main__":
    main()
