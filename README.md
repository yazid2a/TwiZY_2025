# Traffic Sign Detection - Java OpenCV + VLCJ

Ce projet permet de détecter les panneaux de signalisation en temps réel sur une vidéo lue avec VLCJ.

## Prérequis

- Java JDK 8 ou supérieur
- OpenCV 2.4.13 pour Java (lib/opencv-2413.jar et native_libs/opencv_java2413.dll)
- VLC installé sur la machine (pour VLCJ)
- Les fichiers JAR suivants dans le dossier `lib/` :
  - `vlcj-4.x.x.jar`
  - `jna.jar`
  - `jna-platform.jar`
  - `opencv-2413.jar`
- Les images de référence des panneaux dans `reference_signs/` (nommées `ref_XX.jpg` ou `ref_XX.png`)
- Une vidéo de test (ex: `video1.avi`) à la racine du projet

## Structure du projet

```
.
├── src/
│   └── TrafficSignDetector.java
├── lib/
│   ├── opencv-2413.jar
│   ├── vlcj-4.x.x.jar
│   ├── jna.jar
│   └── jna-platform.jar
├── native_libs/
│   └── opencv_java2413.dll
├── reference_signs/
│   └── ref_XX.jpg
├── video1.avi
└── README.md
```

## Compilation

Sous Windows :
```bash
javac -cp ".;lib/*" -d . src/TrafficSignDetector.java
```
Sous Linux/Mac :
```bash
javac -cp ".:lib/*" -d . src/TrafficSignDetector.java
```

## Exécution

Sous Windows :
```bash
java -cp ".;lib/*" TrafficSignDetector
```
Sous Linux/Mac :
```bash
java -cp ".:lib/*" TrafficSignDetector
```

## Utilisation

- Placez vos images de référence dans `reference_signs/` (ex: `ref_30.jpg`, `ref_50.jpg`...)
- Placez votre vidéo de test à la racine du projet (ex: `video1.avi`)
- Lancez le programme :
  - La vidéo originale s'affiche à gauche
  - La vidéo traitée (avec détection de panneaux) s'affiche à droite
  - Le panneau détecté s'affiche en bas

## Dépannage

- **Aucun panneau détecté** :
  - Vérifiez la qualité et le nommage des images de référence
  - Utilisez des images nettes, bien cadrées, taille ≥ 64x64 px
- **Erreur VLCJ/JNA** :
  - Vérifiez la présence des JAR dans `lib/` et le classpath
- **Erreur OpenCV** :
  - Vérifiez la présence de la DLL dans `native_libs/` et le paramètre `-Djava.library.path` si besoin
- **Erreur vidéo** :
  - Vérifiez que la vidéo est lisible par VLC
  - Essayez de convertir la vidéo en MJPEG ou XVID si besoin

## Auteur
- Projet adapté et automatisé par IA (OpenAI GPT-4) 