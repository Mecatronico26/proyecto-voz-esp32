import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse

# Configuración
SAMPLE_RATE = 16000
DURATION = 1.0  # 1 segundo
MFCC_NUM = 13
FRAMES = 40
CLASSES = ['adelante', 'atras', 'derecha', 'izquierda', 'ruido']

def load_audio(file_path):
    # Cargar el audio y asegurarse de que tenga la duración correcta
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)), 'constant')
    # Calcular MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=SAMPLE_RATE, 
        n_mfcc=MFCC_NUM, 
        n_fft=512, 
        hop_length=int(SAMPLE_RATE / FRAMES)
    )
    return mfcc.T  # Forma: (FRAMES, MFCC_NUM)

def load_dataset(dataset_path):
    data, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(dataset_path, cls)
        for file in os.listdir(cls_dir):
            if file.endswith('.wav'):
                mfcc = load_audio(os.path.join(cls_dir, file))
                data.append(mfcc)
                labels.append(idx)
    return np.array(data), np.array(labels)

def augment_audio(audio):
    # Ruido sintético
    noise = np.random.normal(0, 0.005, audio.shape)
    audio_noisy = audio + noise
    # Desplazamiento de pitch (solo para audios completos)
    if len(audio) == SAMPLE_RATE:
        audio_pitched = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=2)
    else:
        audio_pitched = audio  # No aplicar si es corto
    # Cambio de velocidad (solo para audios completos)
    if len(audio) == SAMPLE_RATE:
        audio_speed = librosa.effects.time_stretch(audio, rate=1.1)
    else:
        audio_speed = audio  # No aplicar si es corto
    return [audio_noisy, audio_pitched, audio_speed]

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(dataset_path, model_path, epochs=50):
    # Cargar dataset
    X, y = load_dataset(dataset_path)
    
    # Dividir dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    # Data augmentation (solo en entrenamiento)
    X_train_aug, y_train_aug = [], []
    for mfcc, label in zip(X_train, y_train):
        X_train_aug.append(mfcc)
        y_train_aug.append(label)
        # Aplicar augment_audio solo si mfcc tiene la forma correcta
        if mfcc.shape == (FRAMES, MFCC_NUM):
            for aug in augment_audio(mfcc.flatten()):
                aug_reshaped = aug.reshape(FRAMES, MFCC_NUM)
                X_train_aug.append(aug_reshaped)
                y_train_aug.append(label)
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    # Construir y entrenar modelo
    input_shape = (FRAMES, MFCC_NUM, 1)
    model = build_model(input_shape, len(CLASSES))
    model.fit(X_train_aug[..., np.newaxis], y_train_aug, epochs=epochs, validation_data=(X_val[..., np.newaxis], y_val))
    
    # Evaluar modelo
    test_loss, test_acc = model.evaluate(X_test[..., np.newaxis], y_test)
    print(f'Precisión en test: {test_acc:.2f}')
    
    # Exportar modelo a .tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Modelo guardado en: {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar un modelo de reconocimiento de voz')
    parser.add_argument('--dataset_path', type=str, required=True, help='Ruta al dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta donde guardar el modelo')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    args = parser.parse_args()
    main(args.dataset_path, args.model_path, args.epochs)
    