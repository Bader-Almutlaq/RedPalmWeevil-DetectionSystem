#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "esp_camera.h"

#define LED_PIN 2    // Onboard LED pin
#define MODE_PIN 5   // Mode pin: HIGH = Viewer Mode, LOW = Collector Mode

// Maximum number of stored DataBlocks
#define MAX_ENTRIES 10  
volatile bool signalReceived = false;  // Flag for received data

// DataBlock structure for storing data
struct DataBlock {
  uint8_t Device_MAC[6];    // Sender's MAC address
  float GPS[2];             // GPS coordinates [latitude, longitude]
  float ML_classification;  // ML classification result
  camera_fb_t *fb;          // Image frame buffer
};

// Array-based storage for received DataBlocks (acting as a hash table)
DataBlock hashTable[MAX_ENTRIES];

// Track the number of received DataBlocks
int dataCount = 0;

// Device MAC address
uint8_t Device_MAC[6];

// Utility function to convert MAC address to a string
void printMAC(const uint8_t *mac) {
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", mac[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.println();
}

// Print hash table contents
void printHashTable() {
  Serial.println("---- Hash Table Contents ----");
  for (int i = 0; i < MAX_ENTRIES; i++) {
    if (hashTable[i].Device_MAC[0] != 0) { // If not empty
      Serial.print("Entry "); Serial.print(i); Serial.print(" | Device MAC: ");
      printMAC(hashTable[i].Device_MAC);
      Serial.print("   GPS: ");
      Serial.print(hashTable[i].GPS[0], 6);
      Serial.print(", ");
      Serial.print(hashTable[i].GPS[1], 6);
      Serial.print("   ML: ");
      Serial.println(hashTable[i].ML_classification, 2);
      if (hashTable[i].fb != nullptr) {
        Serial.print("   Image size: ");
        Serial.println(hashTable[i].fb->len);
      }
    }
  }
  Serial.println("-----------------------------");
}

// ESP-NOW reception callback (stores data in hashTable)
void onDataRecv(const esp_now_recv_info *info, const uint8_t *data, int len) {
  signalReceived = true; // Indicate data was received
  if (len != sizeof(DataBlock)) {
    Serial.println("Received invalid DataBlock size!");
    return;
  }

  if (dataCount >= MAX_ENTRIES) {
    Serial.println("Storage full! Ignoring new data.");
    return;
  }

  // Store the received DataBlock
  memcpy(&hashTable[dataCount], data, len);
  dataCount++;

  Serial.println("DataBlock received and stored!");
  Serial.print("From MAC: "); printMAC(info->src_addr);
  Serial.print("GPS: "); Serial.print(hashTable[dataCount - 1].GPS[0]);
  Serial.print(", "); Serial.println(hashTable[dataCount - 1].GPS[1]);
  Serial.print("ML Classification: "); Serial.println(hashTable[dataCount - 1].ML_classification);

}

// Set up ESP‑NOW (Collector/Drone Node Mode)
void setupESPNow() {
  WiFi.mode(WIFI_STA);
  esp_wifi_get_mac(WIFI_IF_STA, Device_MAC);

  Serial.print("Device MAC: ");
  printMAC(Device_MAC);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  // Add broadcast peer (for sending)
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  uint8_t broadcastMAC[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
  memcpy(peerInfo.peer_addr, broadcastMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add broadcast peer");
  } else {
    Serial.println("Broadcast peer added");
  }

  esp_now_register_recv_cb(onDataRecv);
  Serial.println("ESP-NOW initialized");
}

// Function to send a DataBlock via broadcast
void sendBroadcastDataBlock() {
  DataBlock db;
  uint8_t broadcastMAC[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
  esp_err_t result = esp_now_send(broadcastMAC, (uint8_t *)&db, sizeof(DataBlock));

  if (result == ESP_OK) {
    Serial.println("Broadcast DataBlock sent.");
  } else {
    Serial.print("Error sending DataBlock: ");
    Serial.println(result);
  }
}

void setup() {
  Serial.begin(115200);
  setupESPNow();
  pinMode(LED_PIN, OUTPUT);
  pinMode(MODE_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // Indicate active mode
  digitalWrite(MODE_PIN, LOW);  // Default to Collector Mode
}

void loop() {
  bool isViewerMode = (digitalRead(MODE_PIN) == HIGH);

  if (isViewerMode) {
    // Periodically print the hash table contents (every 10 seconds).
    static unsigned long lastPrintTime = 0;
    if (millis() - lastPrintTime > 10000) {
      printHashTable();
      lastPrintTime = millis();
    }
  } else {
    // Collector Mode: Send broadcast, then listen for 5 seconds
    Serial.println("Sending broadcast...");
    sendBroadcastDataBlock();
    signalReceived = false;  // Reset signal flag

    unsigned long startTime = millis();
    Serial.println("Listening for responses...");

    // Listen for incoming data for 5 seconds
    while (millis() - startTime < 5000) {
      if (signalReceived) {
        Serial.println("Data received, processing...");
        break; // Exit early if data is received
      }
    }

    if (!signalReceived) {
      Serial.println("No response received in 5 seconds.");
    }
    
    delay(2000);  // Small delay before next broadcast cycle
  }
}
