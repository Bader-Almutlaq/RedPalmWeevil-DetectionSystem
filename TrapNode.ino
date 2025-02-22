#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include "esp_camera.h"

// Pin configuration for Freenove ESP32-Cam WROVER
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  21
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    19
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    5
#define Y2_GPIO_NUM    4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// Global variables
const float threshold = 0.8;
volatile bool signalReceived = false;
uint8_t Device_MAC[6];  // Store device's MAC address

// Fixed Sender MAC address 2C:BC:BB:0D:72:08
uint8_t Sender_MAC[6] = {0x2C, 0xBC, 0xBB, 0x0D, 0x72, 0x08};

// Define the DataBlock structure
typedef struct {
    uint8_t Device_MAC[6];    // Device MAC address
    float GPS[2];             // GPS coordinates [latitude, longitude]
    float ML_classification;  // ML classification result
    camera_fb_t *fb;          // Pointer to image (frame buffer)
} DataBlock;

// Prints MAC address of a given array
void printMAC(const uint8_t *mac) {
    for (int i = 0; i < 6; i++) {
        Serial.printf("%02X", mac[i]);
        if (i < 5) Serial.print(":");
    }
    Serial.println();
}

// Callback function when data is received
void OnDataRecv(const esp_now_recv_info* info, const uint8_t *incomingData, int len) {
    Serial.println("Signal received from sender!");
    signalReceived = true;
    // No need to store the sender's MAC, as it's now fixed.
    Serial.print("Sender MAC: ");
    printMAC(Sender_MAC);
}

// Sets up ESP-NOW and adds the broadcast peer.
void setupESPNow() {
    Serial.print("------------------------------------------\n");
    WiFi.mode(WIFI_STA);
    esp_wifi_get_mac(WIFI_IF_STA, Device_MAC);
    Serial.print("MAC address: ");
    printMAC(Device_MAC);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW Init Failed");
        return;
    }
    
    // Add broadcast peer
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    uint8_t broadcastMAC[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    memcpy(peerInfo.peer_addr, broadcastMAC, 6);
    peerInfo.channel = 0;     // Use current channel
    peerInfo.encrypt = false; // No encryption
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add broadcast peer");
    } else {
        Serial.println("Broadcast peer added.");
    }
    
    esp_now_register_recv_cb(OnDataRecv);
    Serial.println("ESP-NOW Initialized");
    Serial.print("------------------------------------------\n");
}

// Initializes ESP-CAM
void setupCamera() {
    Serial.print("------------------------------------------\n");
    Serial.println("ESP-Cam Initializing...");

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
    config.frame_size = FRAMESIZE_UXGA;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    if (config.pixel_format == PIXFORMAT_JPEG) {
        if (psramFound()) {
            config.jpeg_quality = 10;
            config.fb_count = 2;
            config.grab_mode = CAMERA_GRAB_LATEST;
        } else {
            config.frame_size = FRAMESIZE_SVGA;
            config.fb_location = CAMERA_FB_IN_DRAM;
        }
    } else {
        config.frame_size = FRAMESIZE_240X240;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }

    if (config.pixel_format == PIXFORMAT_JPEG) {
        s->set_framesize(s, FRAMESIZE_QVGA);
    }

    Serial.println("ESP-Cam Initialized Successfully");
    Serial.print("------------------------------------------\n");
}

// Captures an image and returns the framebuffer
camera_fb_t* captureImage() {
    Serial.println("Capturing image...");
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Failed to capture image!");
        return nullptr;
    }
    Serial.printf("Captured image size: %d bytes\n", fb->len);
    return fb;
}

// Sends the DataBlock over ESP-NOW
void sendDataBlock(DataBlock *db) {
    if (!db) {
        Serial.println("No data to send!");
        return;
    }
    Serial.println("Sending DataBlock...");
    esp_now_send(Device_MAC, (uint8_t *)db, sizeof(DataBlock));  // Send the DataBlock as raw data
}

void setup() {
    Serial.begin(115200);
    setupESPNow();
    setupCamera();
}

void loop() {
    Serial.print(millis());
    Serial.println(": Waiting for signal...");
    signalReceived = false;
    unsigned long startTime = millis();

    while (millis() - startTime < 10000) {
        if (signalReceived) {
            camera_fb_t *fb = captureImage();
            if (fb) {
                // Prepare the DataBlock
                DataBlock db;
                memcpy(db.Device_MAC, Device_MAC, 6);  // Store device MAC address
                db.GPS[0] = 25.276987;  // Example latitude
                db.GPS[1] = 55.296249;  // Example longitude
                db.ML_classification = 0.85;  // Example classification result
                if (db.ML_classification >= threshold) {
                    db.fb = fb;  // Assign the captured image (frame buffer)
                } else {
                    db.fb = NULL;  // No image assigned if classification is below threshold
                }
                
                // Send the DataBlock
                sendDataBlock(&db);

                // Free the image frame buffer after sending
                esp_camera_fb_return(fb);
            }
            return;
        }
    }
    Serial.println("No signal received. Capturing another image...\n");
    Serial.print("------------------------------------------\n");
}
