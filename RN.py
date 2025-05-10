import os
import cv2 # OpenCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # Pour le one-hot encoding
# Optionnel, si vous avez un fichier CSV pour les noms de classes
# import pandas as pd

# --- Paramètres à configurer pour GTSRB ---
# MODIFIEZ CECI : Chemin vers le dossier principal de votre jeu de données GTSRB.
# Cela pourrait être le dossier 'Train' à l'intérieur du dossier GTSRB téléchargé,
# ou directement le dossier qui contient les sous-dossiers numérotés des classes.
DATASET_PATH = "Train" # EXEMPLE, À ADAPTER !
# Ou si vous avez un dossier GTSRB qui contient 'Train', 'Test', etc.
# DATASET_PATH = "chemin/vers/votre/GTSRB_dataset_folder/Train"

# Taille à laquelle redimensionner les images (largeur, hauteur)
# Les tailles courantes pour GTSRB sont 32x32 ou 48x48.
IMG_WIDTH = 48
IMG_HEIGHT = 48

# Nombre total de classes pour GTSRB
NUM_CLASSES = 43

# Optionnel: Chemin vers le fichier CSV qui mappe les ID de classe aux noms (si disponible)
# CLASS_NAMES_CSV_PATH = "chemin/vers/votre/GTSRB/signnames.csv" # EXEMPLE

# --- Chargement et Prétraitement ---
def load_gtsrb_data(dataset_path, img_width, img_height, num_classes):
    images = []
    labels = []
    # class_names_map = {} # Si vous chargez les noms depuis un CSV

    # Optionnel: Charger les noms de classes depuis un CSV
    # try:
    #     if CLASS_NAMES_CSV_PATH and os.path.exists(CLASS_NAMES_CSV_PATH):
    #         sign_names_df = pd.read_csv(CLASS_NAMES_CSV_PATH)
    #         for _, row in sign_names_df.iterrows():
    #             class_names_map[int(row['ClassId'])] = row['SignName']
    #         print(f"Noms des classes chargés depuis : {CLASS_NAMES_CSV_PATH}")
    # except Exception as e:
    #    print(f"Avertissement: Impossible de charger les noms de classes depuis CSV: {e}")


    print(f"Chargement des images depuis : {dataset_path}")

    try:
        # Les dossiers de classe dans GTSRB sont typiquement nommés '0', '1', ..., '42' ou '00000', '00001', ...
        # Nous allons les trier pour s'assurer que l'index correspond à l'ID de la classe.
        class_folders_names = sorted(
            [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))],
            key=lambda x: int(x) # Trie numériquement les noms de dossiers
        )
    except FileNotFoundError:
        print(f"ERREUR: Le dossier du jeu de données '{dataset_path}' n'a pas été trouvé.")
        print("Veuillez vérifier la variable DATASET_PATH.")
        return None, None
    except ValueError:
        print(f"ERREUR: Les noms des sous-dossiers dans '{dataset_path}' ne semblent pas être des entiers.")
        print("Assurez-vous que les dossiers de classe sont nommés numériquement (ex: '0', '1', ..., '42').")
        return None, None


    if not class_folders_names:
        print(f"ERREUR: Aucun sous-dossier de classe trouvé dans '{dataset_path}'.")
        return None, None

    if len(class_folders_names) != num_classes:
        print(f"AVERTISSEMENT: Trouvé {len(class_folders_names)} dossiers de classe, mais NUM_CLASSES est défini sur {num_classes}.")
        print("Pour GTSRB, cela devrait être 43. Vérifiez votre dossier DATASET_PATH.")
        # Vous pourriez vouloir arrêter ici si la disparité est grande.

    actual_class_ids_found = []

    for class_folder_name in class_folders_names:
        try:
            class_id = int(class_folder_name) # Le nom du dossier est l'ID de la classe
            actual_class_ids_found.append(class_id)
        except ValueError:
            print(f"  Avertissement: Le nom du dossier '{class_folder_name}' n'est pas un entier valide. Ignoré.")
            continue

        class_path = os.path.join(dataset_path, class_folder_name)
        print(f"Chargement de la classe : {class_folder_name} (ID: {class_id})")
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg', '.ppm'))]

        if not image_files:
            print(f"  Avertissement: Aucun fichier image trouvé dans {class_path}")
            continue

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"    Avertissement : Impossible de lire l'image {img_path}")
                    continue

                img_resized = cv2.resize(img, (img_width, img_height))
                images.append(img_resized)
                labels.append(class_id)
            except Exception as e:
                print(f"    Erreur lors du traitement de l'image {img_path}: {e}")

    if not images:
        print("ERREUR: Aucune image n'a été chargée. Vérifiez la structure de votre jeu de données et les chemins.")
        return None, None

    images_np = np.array(images, dtype="float32")
    labels_np = np.array(labels)

    images_np /= 255.0 # Normalisation

    # S'assurer que tous les labels sont dans la plage [0, num_classes-1]
    if np.min(labels_np) < 0 or np.max(labels_np) >= num_classes:
        print(f"ERREUR: Les labels des classes (min: {np.min(labels_np)}, max: {np.max(labels_np)}) sont en dehors de la plage attendue [0, {num_classes-1}].")
        print("Vérifiez la structure de vos dossiers de classe et NUM_CLASSES.")
        # Vous pouvez imprimer les actual_class_ids_found pour déboguer
        print(f"IDs de classe trouvés dans les noms de dossiers: {sorted(list(set(actual_class_ids_found)))}")
        return None, None


    labels_one_hot = to_categorical(labels_np, num_classes=num_classes)

    print(f"\nChargement terminé.")
    print(f"Nombre total d'images chargées : {len(images_np)}")
    print(f"Forme du tableau d'images : {images_np.shape}")
    print(f"Forme du tableau de labels (one-hot) : {labels_one_hot.shape}")
    print(f"Nombre de classes uniques dans les labels chargés : {len(np.unique(labels_np))}")

    return images_np, labels_one_hot

# --- Appel de la fonction ---
images_data, labels_data = load_gtsrb_data(DATASET_PATH, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES)

if images_data is not None and labels_data is not None:
    print("\nAffichage de quelques images prétraitées avec leurs labels...")
    plt.figure(figsize=(12, 12))
    num_examples_to_show = min(16, len(images_data))
    for i in range(num_examples_to_show):
        plt.subplot(4, 4, i + 1)
        # OpenCV charge en BGR, Matplotlib s'attend à RGB. Conversion pour affichage correct.
        plt.imshow(cv2.cvtColor(images_data[i], cv2.COLOR_BGR2RGB))
        original_label_index = np.argmax(labels_data[i]) # Récupère l'index de la classe depuis one-hot
        # Si vous avez chargé class_names_map :
        # title = class_names_map.get(original_label_index, f"Classe: {original_label_index}")
        # plt.title(title)
        # Sinon, affichez simplement l'index :
        plt.title(f"Classe ID: {original_label_index}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # --- Division des Données (Training / Validation) ---
    X_train, X_val, y_train, y_val = train_test_split(
        images_data, labels_data,
        test_size=0.2, # 20% pour la validation
        random_state=42,
        stratify=labels_data # Important pour la distribution des classes
    )

    print(f"\nDivision des données terminée.")
    print(f"Forme de X_train (images d'entraînement) : {X_train.shape}")
    print(f"Forme de y_train (labels d'entraînement) : {y_train.shape}")
    print(f"Forme de X_val (images de validation) : {X_val.shape}")
    print(f"Forme de y_val (labels de validation) : {y_val.shape}")

else:
    print("Le chargement des données GTSRB a échoué. Veuillez vérifier les messages d'erreur.")

# Assurez-vous que ces variables de la section précédente sont disponibles :
# X_train, y_train, X_val, y_val, NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Étape 2.D : Définition du Modèle CNN ---

def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # Couche de Convolution 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25)) # Optionnel: Dropout après pooling

    # Couche de Convolution 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25)) # Optionnel

    # Couche de Convolution 3 (Optionnelle, pour plus de profondeur)
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25)) # Optionnel

    # Aplatir les features pour les couches denses
    model.add(Flatten())

    # Couche Dense (entièrement connectée)
    model.add(Dense(512, activation='relu')) # Un nombre plus élevé de neurones ici peut aider
    model.add(Dropout(0.5)) # Dropout important avant la couche de sortie pour régulariser

    # Couche de Sortie
    model.add(Dense(num_classes, activation='softmax')) # Softmax pour la classification multi-classes

    return model

# Définir la forme d'entrée pour le modèle
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3) # (hauteur, largeur, canaux)

# Créer le modèle
model = create_cnn_model(input_shape, NUM_CLASSES)

# Afficher le résumé du modèle
model.summary()

# --- Compilation du Modèle ---
# L'optimiseur Adam est un bon choix par défaut.
# La learning rate peut être ajustée si besoin.
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', # Pour labels one-hot
              metrics=['accuracy'])

# --- Callbacks (Optionnel mais recommandé) ---
# Sauvegarder le meilleur modèle basé sur la 'val_accuracy'
# MODIFIEZ CECI : Choisissez un nom de fichier pour votre modèle sauvegardé
MODEL_SAVE_PATH = "gtsrb_cnn_model_best.keras" # Utiliser l'extension .keras (nouveau format Keras) ou .h5 (ancien format)

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

# Arrêter l'entraînement si 'val_loss' ne s'améliore pas pendant 'patience' époques
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10, # Nombre d'époques sans amélioration avant d'arrêter
                               verbose=1,
                               mode='min',
                               restore_best_weights=True) # Restaure les poids du meilleur epoch

callbacks_list = [checkpoint, early_stopping]
# Si vous ne voulez pas d'arrêt anticipé au début, vous pouvez commencer avec :
# callbacks_list = [checkpoint]


# --- Entraînement du Modèle ---
EPOCHS = 30 # Nombre d'époques. Peut être augmenté si EarlyStopping n'est pas atteint.
BATCH_SIZE = 64 # Taille du lot. 32, 64, 128 sont des valeurs courantes.

print("\nDébut de l'entraînement du modèle...")

history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list, # Ajouter les callbacks ici
                    verbose=1) # verbose=1 pour voir la progression

print("\nEntraînement terminé.")

# --- Évaluation du Modèle (sur l'ensemble de validation) ---
# Si EarlyStopping a restauré les meilleurs poids, cette évaluation sera sur le meilleur modèle.
# Sinon, ce sera sur le dernier état du modèle.
print("\nÉvaluation du modèle sur les données de validation...")
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Perte de validation (Validation Loss) : {val_loss:.4f}")
print(f"Précision de validation (Validation Accuracy) : {val_accuracy:.4f} ({(val_accuracy*100):.2f}%)")

# --- Visualisation de l'Apprentissage ---
print("\nAffichage des courbes d'apprentissage...")

plt.figure(figsize=(12, 4))

# Graphique de la Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision Validation')
plt.title('Précision du Modèle')
plt.ylabel('Précision')
plt.xlabel('Époque')
plt.legend(loc='lower right')

# Graphique de la Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte Entraînement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.title('Perte du Modèle')
plt.ylabel('Perte')
plt.xlabel('Époque')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print(f"\nLe meilleur modèle a été sauvegardé (si ModelCheckpoint a été utilisé) dans : {MODEL_SAVE_PATH}")
print("Si vous n'avez pas utilisé ModelCheckpoint, le modèle actuel (dernier état) est en mémoire.")
print("Vous pouvez le sauvegarder manuellement avec : model.save('mon_modele_final.keras')")

