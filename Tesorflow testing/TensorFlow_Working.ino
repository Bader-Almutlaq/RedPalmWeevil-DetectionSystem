#include "esp_camera.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <esp_heap_caps.h>
#include "SD_MMC.h"
#include "FS.h"

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 21
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 19
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 5
#define Y2_GPIO_NUM 4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#define SD_MMC_CMD 15
#define SD_MMC_CLK 14
#define SD_MMC_D0 2

#define IMG_WIDTH 96
#define IMG_HEIGHT 96
#define MODEL_CHANNELS 3


const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = (300 * 1024);
uint8_t* tensor_arena = nullptr;


void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QQVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;


  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (1)
      ;
  }

  sensor_t* s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_96X96);  // force 96x96 resolution after init
  s->set_vflip(s, 1);                    // vertical flip often needed
  s->set_brightness(s, 1);               // optional tweak
  s->set_saturation(s, -2);              // optional tweak
}

void setupSDCard() {
  SD_MMC.setPins(SD_MMC_CLK, SD_MMC_CMD, SD_MMC_D0);

  if (!SD_MMC.begin("/sdcard", true, true, SDMMC_FREQ_DEFAULT, 5)) {
    Serial.println("Card Mount Failed");
    return;
  }

  uint8_t cardType = SD_MMC.cardType();
  if (cardType == CARD_NONE) {
    Serial.println("No SD_MMC card attached");
    return;
  }

  Serial.print("SD_MMC Card Type: ");
  if (cardType == CARD_MMC) {
    Serial.println("MMC");
  } else if (cardType == CARD_SD) {
    Serial.println("SDSC");
  } else if (cardType == CARD_SDHC) {
    Serial.println("SDHC");
  } else {
    Serial.println("UNKNOWN");
  }

  uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
  Serial.printf("SD_MMC Card Size: %lluMB\n", cardSize);
}

void saveImage(camera_fb_t* fb, unsigned long trap_id) {
  String filename = "/" + String(trap_id) + ".jpg";
  fs::FS& fs = SD_MMC;
  File file = fs.open(filename.c_str(), FILE_WRITE);
  if (!file) {
    Serial.printf("Failed to open file in writing mode");
  } else {
    file.write(fb->buf, fb->len);  // payload (image), payload length
    Serial.printf("Saved: %s\n", filename.c_str());
  }
  file.close();
  esp_camera_fb_return(fb);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  setupCamera();
  setupSDCard();

  if (!psramFound()) {
    Serial.println("❌ PSRAM not found!");
    while (1)
      ;
  }

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided and schema version are not equal!");
    while (true)
      ;  // stop program here
  }

  static tflite::AllOpsResolver resolver;

  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena");
    while (1)
      ;
  }

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true)
      ;  // stop program here
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete. Starting inference...");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();  // remove whitespace/newlines

    if (cmd.equalsIgnoreCase("go")) {
      Serial.println("Running inference...");

      camera_fb_t* fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("Camera capture failed");
        return;
      }

      Serial.print("Image captured: ");
      Serial.print(fb->len);
      Serial.println(" bytes");

      if (fb->width != IMG_WIDTH || fb->height != IMG_HEIGHT) {
        Serial.println("Unexpected image size");
        esp_camera_fb_return(fb);
        return;
      }

      saveImage(fb, millis());

      // Allocate RGB buffer
      uint8_t* rgb_buf = (uint8_t*)malloc(IMG_WIDTH * IMG_HEIGHT * MODEL_CHANNELS);
      if (!rgb_buf) {
        Serial.println("Failed to allocate RGB buffer");
        esp_camera_fb_return(fb);
        return;
      }

      // Decode JPEG to RGB888
      if (!fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_buf)) {
        Serial.println("JPEG to RGB888 conversion failed");
        free(rgb_buf);
        esp_camera_fb_return(fb);
        return;
      }

      // Get model input tensor
      TfLiteTensor* input = interpreter->input(0);
      int8_t* inputBuffer = input->data.int8;

      // Quantize: input = uint8 - 128
      for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT * MODEL_CHANNELS; i++) {
        inputBuffer[i] = (int8_t)rgb_buf[i] - 128;
      }
      // for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
      //   input->data.int8[i] = (int8_t)fb->buf[i] - 128;
      // }

      esp_camera_fb_return(fb);

      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
      }

      // TfLiteTensor* output = interpreter->output(0);

      // // Obtain the quantized output from model's output tensor
      // uint8_t quan_not_rpw = output->data.int8[0];
      // uint8_t quan_rpw = output->data.int8[1];

      // // Dequantize the output from integer to floating-point
      // float not_rpw_score = (quan_not_rpw - output->params.zero_point) * output->params.scale;
      // float rpw_score = (quan_rpw - output->params.zero_point) * output->params.scale;


      // // dequan = (output - zero_point) * scale
      // Serial.printf("NO-RPW: %d, score: %f\n", quan_not_rpw, not_rpw_score);
      // Serial.printf("RPW: %d, score: %f\n", quan_rpw, rpw_score);
      // Assume output is int8 and quantized (adjust if float model)
      int8_t* output_data = output->data.int8;
      float scale = output->params.scale;
      int zero_point = output->params.zero_point;

      int num_classes = output->bytes;  // e.g., 2 for binary, 3+ for multi-class

      int max_index = 0;
      float max_score = -999;

      Serial.println("Class scores:");
      for (int i = 0; i < num_classes; i++) {
        float score = (output_data[i] - zero_point) * scale;
        Serial.printf("  Class %d: %.4f\n", i, score);
        if (score > max_score) {
          max_score = score;
          max_index = i;
        }
      }

      Serial.printf("🔍 Predicted class: %d (score: %.4f)\n", max_index, max_score);


    } else {
      Serial.println("Unknown command. Type 'go' to capture and classify.");
    }
  }
}