# Sistem Anti-Plagiat

Un sistem de monitorizare pentru prevenirea plagiatului in timpul examenelor, implementat in Python cu PyQt5.

## Functionalitati

- Monitorizarea in timp real a candidatului
- Detectarea directiei privirii
- Detectarea obiectelor interzise (telefoane, smartwatch-uri)
- Inregistrare video
- Generare rapoarte de incalcari

## Instalare

1. Cloneaza repository-ul
2. Instaleaza dependentele: `pip install -r requirements.txt`
3. Descarca modelul facial: `shape_predictor_68_face_landmarks.dat`
4. Ruleaza aplicatia: `python gui_app.py`

## Configurare

Editeaza fisierul `config.json` pentru a personaliza setarile.