import numpy as np
import tensorflow as tf
import librosa
import os

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=16000, duration=1.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=400)
    return mfcc.T  # (40, 13)

def main(test_dir, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for cls in os.listdir(test_dir):
        cls_dir = os.path.join(test_dir, cls)
        for file in os.listdir(cls_dir):
            if file.endswith('.wav'):
                mfcc = load_audio(os.path.join(cls_dir, file))
                mfcc = np.expand_dims(mfcc, axis=0)  # (1, 40, 13)
                mfcc = np.expand_dims(mfcc, axis=-1)  # (1, 40, 13, 1)
                interpreter.set_tensor(input_details[0]['index'], mfcc.astype(np.float32))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(output)
                print(f"Archivo: {file}, Clase real: {cls}, Predicción: {predicted_class}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='modelo_comandos.tflite')
    args = parser.parse_args()
    main(args.test_dir, args.model_path)