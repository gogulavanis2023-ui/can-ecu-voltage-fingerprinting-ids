#include <SPI.h>
#include <mcp_can.h>

#define CS_PIN   10
#define INT_PIN  2

MCP_CAN CAN0(CS_PIN);

void setup() {
  Serial.begin(115200);
  while (CAN0.begin(MCP_ANY, CAN_125KBPS, MCP_8MHZ) != CAN_OK) {
    Serial.println("MCP2515 init FAILED - check SPI wiring / power");
    delay(1000);
  }
  Serial.println("MCP2515 init OK - listening...");
  CAN0.setMode(MCP_NORMAL);
  pinMode(INT_PIN, INPUT);
}

void loop() {
  if (!digitalRead(INT_PIN)) {
    unsigned long rxId;
    byte len;
    byte buf[8];
    CAN0.readMsgBuf(&rxId, &len, buf);

    Serial.print("Got ID 0x");
    Serial.print(rxId & 0x7FF, HEX);
    Serial.print("  len=");
    Serial.print(len);
    Serial.print("  data:");
    for (byte i = 0; i < len; i++) {
      Serial.print(" ");
      Serial.print(buf[i]);
    }
    Serial.println();
  }
}