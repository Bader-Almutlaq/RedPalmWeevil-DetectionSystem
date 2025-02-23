#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "esp_camera.h"

#define LED_PIN 2   // Onboard LED pin
#define MODE_PIN 5  // Mode pin: HIGH = Viewer Mode, LOW = Collector Mode

// Maximum number of stored DataBlocks
#define MAX_ENTRIES 10
volatile bool signalReceived = false;  // Flag for received data

// Drone MAC address 2C:BC:BB:0D:72:08
// Trap  MAC address CC:7B:5C:98:67:50

//Note: trap MAC must be given, but drone MAC can be obtained by esp_wifi_get_mac()
uint8_t Drone_MAC[6];
uint8_t Trap_MAC[6] = { 0xCC, 0x7B, 0x5C, 0x98, 0x67, 0x50 };

// Structure for receiving data
typedef struct  {
  float ML_classification;  // ML classification result
  float lat;                // GPS coordinates latitude
  float lng;                // GPS coordinates longitude
  //camera_fb_t *fb;          // Image frame buffer
}DataBlock;

// Array-based storage for received DataBlocks (acting as a hash table)
DataBlock hashTable[MAX_ENTRIES];

// Track the number of received DataBlocks
int dataCount = 0;

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
  // for (int i = 0; i < MAX_ENTRIES; i++) {
  //   if (hashTable[i].Trap_MAC[0] != 0) {  // If not empty
  //     Serial.print("Entry ");
  //     Serial.print(i);
  //     Serial.print(" | Device MAC: ");
  //     printMAC(hashTable[i].Trap_MAC);
  //     Serial.print("   GPS: ");
  //     Serial.print(hashTable[i].GPS[0], 6);
  //     Serial.print(", ");
  //     Serial.print(hashTable[i].GPS[1], 6);
  //     Serial.print("   ML: ");
  //     Serial.println(hashTable[i].ML_classification, 2);
  //     if (hashTable[i].fb != nullptr) {
  //       Serial.print("   Image size: ");
  //       Serial.println(hashTable[i].fb->len);
  //     }
  //   }
  // }
  Serial.println("-----------------------------");
}

// Callback function for receiving data, triggers when trap sends data
// stores data in hashTable
void onDataRecv(const esp_now_recv_info *info, const uint8_t *data, int len) {
  signalReceived = true;  // Indicate data was received

  if (len != sizeof(DataBlock)) {
    Serial.println("Received invalid DataBlock size!");
    return;
  }

  if (dataCount >= MAX_ENTRIES) {
    Serial.println("Storage full! Ignoring new data.");
    return;
  }

  // Store the received DataBlock
  // memcpy(&hashTable[dataCount], data, len);
  // dataCount++;
  DataBlock db;
  memcpy(&db, data, sizeof(data));
  Serial.println("Data received from Trap and stored!");

  // Print the received data from trap
  Serial.print("From MAC: ");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", info->src_addr[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.printf("\nReceived GPS: %f, %f\n", db.lat, db.lng);
  Serial.printf("Received ML classification: %f\n", db.ML_classification);
  // Serial.print("GPS: ");
  // Serial.print(hashTable[dataCount - 1].GPS[0]);
  // Serial.print(", ");
  // Serial.println(hashTable[dataCount - 1].GPS[1]);
  // Serial.print("ML Classification: ");
  // Serial.println(hashTable[dataCount - 1].ML_classification);
}

// Set up ESP‑NOW (Collector/Drone Node Mode)
void setupESPNow() {
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }
  esp_wifi_get_mac(WIFI_IF_STA, Drone_MAC);

  Serial.print("Drone MAC Address: ");
  printMAC(Drone_MAC);

  // add peer info
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, Trap_MAC, 6);
  peerInfo.channel = 0;      // Use current channel
  peerInfo.encrypt = false;  // No encryption
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add trap as a peer");
  } else {
    Serial.println("Trap peer added successfully.");
  }

  esp_now_register_recv_cb(onDataRecv);
  Serial.println("ESP-NOW initialized");
}

// send request to trap, sent data is irrelevant
void send_request_to_trap() {
  char requestMessage[] = "clear_to_send";
  esp_err_t result = esp_now_send(Trap_MAC, (uint8_t *)&requestMessage, sizeof(requestMessage));

  if (result == ESP_OK) {
    Serial.println("Request sent.");
  } else {
    Serial.print("Error sending reqest: ");
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
  }

  else {
    // Collector Mode: Sends requests -> listen for 5 seconds
    Serial.println("Sending request...");
    send_request_to_trap();
    signalReceived = false;  // Reset signal flag

    unsigned long startTime = millis();
    Serial.println("Listening for responses...");
    Serial.print("Drone MAC Address: ");
    printMAC(Drone_MAC);


    // Listen for incoming data for 5 seconds
    while (millis() - startTime < 5000) {
      if (signalReceived) {
        Serial.println("Data received and saved !");
        break;  // Exit early if data is received
      }
    }

    if (!signalReceived) {
      Serial.println("No response received in 5 seconds.");
    }

    delay(2000);  // Small delay before next broadcast cycle
  }
}
