FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir \
    tensorflow==2.18.0 \
    librosa \
    scikit-learn \
    numpy

# Mantener actualizado pip
RUN pip install --upgrade pip

# Crear directorios
RUN mkdir /app /dataset

# Copiar scripts
COPY audios.py /app/
COPY RedNeuronal_Español.py /app/
COPY test_model.py /app/

# Establecer directorio de trabajo
WORKDIR /app

# Comando por defecto (entrenamiento)
CMD ["python", "RedNeuronal_Español.py", "--dataset_path", "/dataset"]