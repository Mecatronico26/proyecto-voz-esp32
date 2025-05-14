import sounddevice as sd
import soundfile as sf
import os

# CONFIGURACIÓN
PALABRAS = ["adelante", "atras", "izquierda", "derecha"]  # Palabras a grabar
CARPETA_SALIDA = "dataset"  # Carpeta donde se guardarán los audios
MUESTRAS_POR_PALABRA = 20  # Número de muestras por palabra
DURACION = 1  # Duración de cada grabación (en segundos)
FS = 16000  # Frecuencia de muestreo (16kHz)

# Crear la carpeta de salida si no existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Grabar y guardar las muestras para cada palabra
for palabra in PALABRAS:
    # Crear carpeta para cada palabra si no existe
    carpeta_palabra = os.path.join(CARPETA_SALIDA, palabra)
    os.makedirs(carpeta_palabra, exist_ok=True)
    print(f"\n🎙️ GRABANDO: {palabra.upper()}")

    for i in range(MUESTRAS_POR_PALABRA):
        # Pedir al usuario que presione ENTER para empezar a grabar
        input(f"Presiona ENTER para grabar muestra {i}/{MUESTRAS_POR_PALABRA} para la palabra {palabra.upper()}")
        
        # Grabar audio
        print("🔴 Grabando...")
        audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='int16')
        sd.wait()  # Esperar a que termine la grabación
        print(f"✅ Grabación {i+1} de '{palabra}' completada.")
        
        # Guardar el archivo
        nombre_archivo = os.path.join(carpeta_palabra, f"{palabra}_{i}.wav")
        sf.write(nombre_archivo, audio, FS)
        print(f"✅ Guardado: {nombre_archivo}")
print("\n🎉 Proceso completado con éxito.")
