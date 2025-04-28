# Sistem Anti-Plagiat

Un sistem de monitorizare pentru prevenirea plagiatului in timpul examenelor, implementat in Python cu PyQt5.

## Functionalitati

- **Monitorizare în timp real**: Urmărește activitatea candidatului folosind camera web.
- **Detectarea direcției privirii**: Utilizează algoritmi de urmărire a privirii pentru a identifica direcția în care se uită candidatul.
- **Detectarea obiectelor interzise**: Recunoaște obiecte precum telefoane mobile sau smartwatch-uri în cadrul video.
- **Înregistrare video**: Salvează fluxul video pentru o analiză ulterioară.
- **Generare rapoarte**: Creează rapoarte detaliate despre încălcările detectate, inclusiv timestamp-uri și descrieri.
- **Export date**: Permite exportarea datelor în formate utilizabile pentru arhivare sau analiză ulterioară.

## Instalare

1. Cloneaza repository-ul
2. Instaleaza dependentele: `pip install -r requirements.txt`
3. Descarca modelul facial: `shape_predictor_68_face_landmarks.dat`
4. Ruleaza aplicatia: `python gui_app.py`

## Configurare

Editeaza fisierul `config.json` pentru a personaliza setarile.