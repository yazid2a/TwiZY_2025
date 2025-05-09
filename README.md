# Projet de Détection de Panneaux de Signalisation

Ce projet utilise OpenCV 2.4.13 pour la détection de panneaux de signalisation en temps réel.

## Structure du Projet

```
.
├── src/                    # Code source Java
├── lib/                    # Bibliothèques JAR (OpenCV)
├── native_libs/           # Bibliothèques natives OpenCV
├── reference_signs/       # Images de référence des panneaux
└── README.md              # Ce fichier
```

## Prérequis

1. JDK 8 (ou version compatible avec OpenCV 2.4.13)
2. OpenCV 2.4.13 pour Java
   - Fichier JAR : `opencv-2413.jar`
   - Bibliothèque native : `opencv_java2413.dll` (Windows) / `libopencv_java2413.so` (Linux) / `libopencv_java2413.dylib` (macOS)

## Installation

1. Téléchargez OpenCV 2.4.13 depuis le site officiel
2. Copiez `opencv-2413.jar` dans le dossier `lib/`
3. Copiez la bibliothèque native appropriée dans le dossier `native_libs/`
4. Placez vos images de référence des panneaux dans le dossier `reference_signs/`

## Compilation

```bash
# Windows
javac -cp ".;lib/opencv-2413.jar" src/TrafficSignDetector.java

# Linux/macOS
javac -cp ".:lib/opencv-2413.jar" src/TrafficSignDetector.java
```

## Exécution

```bash
# Windows
java -Djava.library.path="native_libs" -cp ".;lib/opencv-2413.jar" TrafficSignDetector

# Linux/macOS
java -Djava.library.path="native_libs" -cp ".:lib/opencv-2413.jar" TrafficSignDetector
```

## Configuration IDE

### Eclipse
1. Ajoutez `lib/opencv-2413.jar` au Build Path
2. Dans Run Configurations, ajoutez `-Djava.library.path="chemin/absolu/vers/native_libs"` dans VM arguments

### IntelliJ IDEA
1. Ajoutez `lib/opencv-2413.jar` aux dépendances du projet
2. Dans Run Configurations, ajoutez `-Djava.library.path="chemin/absolu/vers/native_libs"` dans VM options

## Dépannage

1. **Erreur de bibliothèque native**
   - Vérifiez que le chemin de `java.library.path` est correct
   - Assurez-vous que la version de la bibliothèque correspond à votre architecture (32/64 bits)

2. **Erreur de classe non trouvée**
   - Vérifiez que le JAR OpenCV est correctement inclus dans le classpath

3. **Erreur de fichier non trouvé**
   - Vérifiez les chemins des fichiers de référence et de la vidéo
   - Assurez-vous que les fichiers existent et sont accessibles 