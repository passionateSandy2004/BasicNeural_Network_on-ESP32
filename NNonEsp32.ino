#include <ESP8266WiFi.h>
#include <Arduino.h>
#include <pgmspace.h>
#include "mnist_model_weights_int.h"  // Contains: const int8_t model_weights[] = { ... }

// WiFi credentials – update these with your network details
const char* ssid     = "Tenda_B4AE50";
const char* password = "admin@123";
WiFiServer server(5000);

// Network architecture definitions
#define INPUT_SIZE 784    // 28 x 28 image flattened
#define HIDDEN_SIZE 32    // Number of neurons in hidden layer
#define OUTPUT_SIZE 10    // 10 output classes (digits 0-9)
#define WEIGHT_SCALE 0.01f

// ----- Activation Functions -----
float relu(float x) {
  return (x > 0) ? x : 0;
}

void softmax(float *input, float *output, int size) {
  float sum = 0.0;
  float maxVal = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > maxVal) {
      maxVal = input[i];
    }
  }
  for (int i = 0; i < size; i++) {
    output[i] = exp(input[i] - maxVal);
    sum += output[i];
  }
  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

// ----- Helper to Fetch Weights from PROGMEM -----
float getWeight(int index) {
  int8_t quantized = pgm_read_byte(&model_weights[index]);
  return ((float)quantized) * WEIGHT_SCALE;
}

// ----- Neural Network Inference Function -----
// Weight layout: Layer 1 (weights: 784*32, biases: 32) then Layer 2 (weights: 32*10, biases: 10)
void runInference(float *input_data, float *output) {
  float hidden[HIDDEN_SIZE];
  float final[OUTPUT_SIZE];
  int idx = 0;

  // Layer 1: Dense (INPUT_SIZE -> HIDDEN_SIZE) + ReLU activation
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    float sum = 0.0;
    for (int i = 0; i < INPUT_SIZE; i++) {
      sum += input_data[i] * getWeight(idx++);
    }
    sum += getWeight(idx++);  // Bias for neuron j
    hidden[j] = relu(sum);
  }

  // Layer 2: Dense (HIDDEN_SIZE -> OUTPUT_SIZE)
  for (int j = 0; j < OUTPUT_SIZE; j++) {
    float sum = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
      sum += hidden[i] * getWeight(idx++);
    }
    sum += getWeight(idx++);  // Bias for output neuron j
    final[j] = sum;
  }

  // Apply softmax to get probability distribution
  softmax(final, output, OUTPUT_SIZE);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP8266 MNIST Inference Server (Binary Protocol) ===");

  // Connect to WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    retry++;
    if (retry > 60) {  // ~30-second timeout
      Serial.println("\nFailed to connect to WiFi. Restarting...");
      ESP.restart();
    }
  }
  Serial.println("\nWiFi connected successfully!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Start the TCP server
  server.begin();
  Serial.println("TCP Server started on port 5000");
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;

  Serial.println("\nClient connected.");

  // Wait until at least 2 bytes (for the count) are available
  while (client.connected() && client.available() < 2) {
    delay(1);
  }
  
  // Read header: number of pairs (2 bytes, big-endian)
  uint16_t numPairs = 0;
  if (client.readBytes((char*)&numPairs, 2) != 2) {
    Serial.println("Failed to read header");
    client.stop();
    return;
  }
  // Convert from network byte order (big-endian) to little-endian
  numPairs = (numPairs >> 8) | (numPairs << 8);
  Serial.printf("Expecting %u nonzero pixel pairs.\n", numPairs);

  // Initialize the input image with zeros
  float test_image[INPUT_SIZE] = {0.0};

  // For each pair, read index (2 bytes) and value (1 byte)
  for (uint16_t i = 0; i < numPairs; i++) {
    uint16_t idx = 0;
    uint8_t val = 0;
    if (client.readBytes((char*)&idx, 2) != 2) break;
    // Convert index from network byte order
    idx = (idx >> 8) | (idx << 8);
    if (client.readBytes((char*)&val, 1) != 1) break;
    if (idx < INPUT_SIZE) {
      // Convert the received byte back to a float in the range [0, 1]
      test_image[idx] = ((float)val) / 255.0f;
    }
  }

  // Run inference on the reconstructed image
  float output[OUTPUT_SIZE];
  runInference(test_image, output);

  // Determine the predicted label (digit) based on the highest probability
  int predicted_label = 0;
  float max_prob = output[0];
  for (int i = 1; i < OUTPUT_SIZE; i++) {
    if (output[i] > max_prob) {
      max_prob = output[i];
      predicted_label = i;
    }
  }

  // Create and send a JSON response with the prediction and probabilities
  String response = "{\"predicted_label\":";
  response += predicted_label;
  response += ",\"probabilities\":[";
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    response += String(output[i], 3);
    if (i < OUTPUT_SIZE - 1) response += ",";
  }
  response += "]}";
  
  client.println(response);
  Serial.println("Sent response:");
  Serial.println(response);

  delay(100);
  client.stop();
  Serial.println("Client disconnected.");
}
