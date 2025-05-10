# Traffic Sign Detection - Java OpenCV + VLCJ + API Python

Ce projet permet de détecter les panneaux de signalisation en temps réel sur une vidéo lue avec VLCJ, en utilisant une API Python (FastAPI + modèle Keras) pour la reconnaissance.

## Prérequis

### Côté Java
- Java JDK 8 ou supérieur
- OpenCV 2.4.13 pour Java (`lib/opencv-2413.jar` et `native_libs/opencv_java2413.dll`)
- VLC installé sur la machine (pour VLCJ)
- Les fichiers JAR suivants dans le dossier `lib/` :  
   * `vlcj-4.x.x.jar`  
   * `jna.jar`  
   * `jna-platform.jar`  
   * `opencv-2413.jar`
   * `json-20231013.jar` (pour le parsing JSON)
- Les images de référence des panneaux dans `reference_signs/` (nommées `ref_XX.jpg` ou `ref_XX.png`)
- Une vidéo de test (ex: `video1.avi` ou `video2.avi`) à la racine du projet

### Côté Python (API)
- Python 3.8+
- Fichier du modèle Keras (`gtsrb_cnn_model_best.keras`)
- Fichier des noms de classes : `signnames.csv`
- Fichier d'API : `api.py`
- Dépendances Python :
  - fastapi
  - uvicorn
  - tensorflow
  - pillow
  - numpy
  - pandas

## Structure du projet

```
.
├── src/
│   └── TrafficSignDetector.java
├── api.py
├── signnames.csv
├── gtsrb_cnn_model_best.keras
├── lib/
│   ├── opencv-2413.jar
│   ├── vlcj-4.x.x.jar
│   ├── jna.jar
│   ├── jna-platform.jar
│   └── json-20231013.jar
├── native_libs/
│   └── opencv_java2413.dll
├── reference_signs/
│   └── ref_XX.jpg
├── video1.avi
├── video2.avi
├── .gitignore
└── README.md
```

## Compilation Java

Sous Windows :
```sh
javac -cp ".;lib/*" -d . src/TrafficSignDetector.java
```
Sous Linux/Mac :
```sh
javac -cp ".:lib/*" -d . src/TrafficSignDetector.java
```

## Exécution Java

Sous Windows :
```sh
java -cp ".;lib/*" TrafficSignDetector
```
Sous Linux/Mac :
```sh
java -cp ".:lib/*" TrafficSignDetector
```

## Lancement de l'API Python

1. Installe les dépendances :
   ```sh
   pip install fastapi uvicorn tensorflow pillow numpy pandas
   ```
2. Lance l'API :
   ```sh
   uvicorn api:app --reload
   ```
3. L'API sera accessible sur [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)

## Fonctionnement

- La vidéo est lue avec VLCJ.
- À chaque frame, la zone rouge (panneau) est détectée (carré vert).
- **Seule l'image du panneau détecté (ROI) est envoyée à l'API Python** pour classification.
- Le nom du dernier panneau détecté reste affiché jusqu'à détection d'un nouveau.
- Les modèles `.pb` et `.onnx` sont exclus du dépôt via `.gitignore`.

## Auteur

- KERRAZI ELYAZID
- ZIDAOUI BADREDDINE

---

Pour toute question ou amélioration, ouvrez une issue sur le [repo GitHub](https://github.com/yazid2a/TwiZY_2025).
