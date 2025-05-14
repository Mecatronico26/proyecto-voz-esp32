#include <Arduino.h>
#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "modelo_comandos_tflite.h"  // Modelo .tflite convertido a .h

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Configuración de pines
#define I2S_WS 22
#define I2S_SD 21
#define I2S_SCK 26
#define LED_ADELANTE 32
#define LED_ATRAS 27
#define LED_DERECHA 33
#define LED_IZQUIERDA 25

// Configuración de I2S
#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define BUFFER_LEN 16000  // 1 segundo a 16kHz
int16_t audio_buffer[BUFFER_LEN];

// Configuración de TFLite Micro
constexpr int tensor_arena_size = 10000;
uint8_t tensor_arena[tensor_arena_size];
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// Variables compartidas entre tareas
volatile bool audio_ready = false;
volatile float audio_data[BUFFER_LEN];

// Configuración de tareas
TaskHandle_t audioTaskHandle = NULL;
TaskHandle_t inferenceTaskHandle = NULL;

// Configuración de sueño profundo
#define DEEP_SLEEP_TIMEOUT 5000  // 5 segundos sin actividad
unsigned long last_activity_time = 0;

// Función para configurar I2S
void i2s_install() {
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_LEN / 8,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };
  i2s_set_pin(I2S_PORT, &pin_config);
}

// Tarea de captura de audio (Core 0)
void audioTask(void *pvParameters) {
  while (1) {
    size_t bytesRead = 0;
    i2s_read(I2S_PORT, (void*)audio_buffer, BUFFER_LEN * sizeof(int16_t), &bytesRead, portMAX_DELAY);
    
    // Normalizar audio a float32 [-1, 1]
    for (int i = 0; i < BUFFER_LEN; i++) {
      audio_data[i] = (float)audio_buffer[i] / 32768.0f;
    }
    
    audio_ready = true;
    last_activity_time = millis();  // Actualizar tiempo de actividad
    vTaskDelay(pdMS_TO_TICKS(100));  // Esperar 100ms
  }
}

// Tarea de inferencia (Core 1)
void inferenceTask(void *pvParameters) {
  while (1) {
    if (audio_ready) {
      // Copiar datos de audio al tensor de entrada
      memcpy(input->data.f, audio_data, BUFFER_LEN * sizeof(float));
      audio_ready = false;
      
      // Ejecutar inferencia
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        Serial.println("Error al ejecutar el modelo");
        continue;
      }
      
      // Obtener salida
      TfLiteTensor* output = interpreter->output(0);
      int palabra = 0;
      float max_score = output->data.f[0];
      for (int i = 1; i < 4; i++) {
        if (output->data.f[i] > max_score) {
          max_score = output->data.f[i];
          palabra = i;
        }
      }
      
      // Controlar LEDs
      digitalWrite(LED_ADELANTE, palabra == 0 ? HIGH : LOW);
      digitalWrite(LED_ATRAS, palabra == 1 ? HIGH : LOW);
      digitalWrite(LED_DERECHA, palabra == 2 ? HIGH : LOW);
      digitalWrite(LED_IZQUIERDA, palabra == 3 ? HIGH : LOW);
      
      // Imprimir resultado (opcional)
      const char* etiquetas[4] = {"adelante", "atras", "derecha", "izquierda"};
      Serial.print("Palabra detectada: ");
      Serial.print(etiquetas[palabra]);
      Serial.print(" (score: ");
      Serial.print(max_score, 4);
      Serial.println(")");
      
      last_activity_time = millis();  // Actualizar tiempo de actividad
    } else {
      // Verificar si ha pasado el tiempo de inactividad
      if (millis() - last_activity_time > DEEP_SLEEP_TIMEOUT) {
        Serial.println("Entrando en sueño profundo...");
        esp_deep_sleep_start();
      }
    }
    vTaskDelay(pdMS_TO_TICKS(100));  // Esperar 100ms
  }
}

void setup() {
  Serial.begin(115200);
  
  // Inicializar pines de LEDs
  pinMode(LED_ADELANTE, OUTPUT);
  pinMode(LED_ATRAS, OUTPUT);
  pinMode(LED_DERECHA, OUTPUT);
  pinMode(LED_IZQUIERDA, OUTPUT);
  
  // Inicializar I2S
  i2s_install();
  i2s_setpin();
  
  // Inicializar TFLite Micro
  model = tflite::GetModel(modelo_comandos_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  
  // Crear tareas
  xTaskCreatePinnedToCore(audioTask, "AudioTask", 4096, NULL, 1, &audioTaskHandle, 0);
  xTaskCreatePinnedToCore(inferenceTask, "InferenceTask", 8192, NULL, 1, &inferenceTaskHandle, 1);
  
  // Configurar temporizador para despertar del sueño profundo
  esp_sleep_enable_timer_wakeup(5 * 1000000);  // 5 segundos
}

void loop() {
  // El loop principal no hace nada, las tareas se manejan en los núcleos
}