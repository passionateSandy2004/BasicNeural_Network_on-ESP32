# Neural Network Inference on ESP8266

This project demonstrates running a trained neural network (MNIST digit classifier) on an ESP8266 microcontroller. The neural network is trained in Python, converted to C-compatible format, and deployed on the ESP8266 to perform real-time digit classification.

## Project Overview

The project consists of three main components:
1. Python-based neural network training and weight conversion
2. ESP8266 firmware for inference
3. Client-server communication over WiFi

## Hardware Requirements

- ESP8266 Generic Module
- USB cable for programming and power
- WiFi network access

## Software Components

### Python Training and Conversion
- `convert.ipynb`: Jupyter notebook for training the neural network and converting weights to C format
- `predict.ipynb`: Jupyter notebook for testing the model
- Generated weight files:
  - `mnist_model_weights.h`: Original weight file
  - `mnist_model_weights_int.h`: Quantized weights for ESP8266

### ESP8266 Firmware
- `NNonEsp32.ino`: Main Arduino sketch implementing:
  - WiFi server setup
  - Neural network inference
  - Client communication handling

## Network Architecture

The implemented neural network has the following structure:
- Input layer: 784 neurons (28x28 MNIST image)
- Hidden layer: 32 neurons with ReLU activation
- Output layer: 10 neurons (digits 0-9) with softmax activation

## Setup Instructions
*Make Sure the folder name is NNonEsp32*
1. **Configure WiFi Settings**
   - Open `NNonEsp32.ino`
   - Update the WiFi credentials:
     ```cpp
     const char* ssid = "YOUR_WIFI_SSID";
     const char* password = "YOUR_WIFI_PASSWORD";
     ```

2. **Upload the Firmware**
   - Open the Arduino IDE
   - Select the ESP8266 board
   - Upload the `NNonEsp32.ino` sketch

3. **Connect to the Server**
   - The ESP8266 will create a TCP server on port 5000
   - Note the IP address displayed in the Serial Monitor

## Communication Protocol

The server expects data in the following format:
1. 2-byte header indicating the number of pixel pairs
2. For each pixel pair:
   - 2 bytes: pixel index
   - 1 byte: pixel value (0-255)

The server responds with a JSON object containing:
- `predicted_label`: The classified digit (0-9)
- `probabilities`: Array of probabilities for each digit

## Performance Considerations

- Weights are quantized to 8-bit integers for memory efficiency
- The model uses a scale factor of 0.01 for weight conversion
- Inference is performed in real-time on the ESP8266

## Dependencies

- ESP8266WiFi library
- Arduino core for ESP8266

## License

[Add your license information here]

## Acknowledgments

- MNIST dataset
- ESP8266 community
- Arduino platform 
