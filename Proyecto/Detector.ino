// Define un arreglo con los pines donde están conectados los 5 LEDs
const int leds[] = {2, 3, 4, 5, 6};

void setup() {
  // Configura cada pin del arreglo como salida
  for (int i = 0; i < 5; i++) {
    pinMode(leds[i], OUTPUT);
  }

  // Inicia la comunicación serial a 9600 baudios
  Serial.begin(9600);
}

void loop() {
  // Si hay datos disponibles en el puerto serial (desde Python)
  if (Serial.available()) {
    char c = Serial.read();  // Lee un carácter enviado

    // Dependiendo del carácter recibido, enciende o apaga un LED
    switch (c) {
      case 'A': digitalWrite(2, HIGH); break;  // Enciende LED A
      case 'E': digitalWrite(3, HIGH); break;  // Enciende LED E
      case 'I': digitalWrite(4, HIGH); break;  // Enciende LED I
      case 'O': digitalWrite(5, HIGH); break;  // Enciende LED O
      case 'U': digitalWrite(6, HIGH); break;  // Enciende LED U

      case 'a': digitalWrite(2, LOW); break;   // Apaga LED A
      case 'e': digitalWrite(3, LOW); break;   // Apaga LED E
      case 'i': digitalWrite(4, LOW); break;   // Apaga LED I
      case 'o': digitalWrite(5, LOW); break;   // Apaga LED O
      case 'u': digitalWrite(6, LOW); break;   // Apaga LED U
    }
  }
}
