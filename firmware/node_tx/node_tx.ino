#include <SPI.h>
#include <mcp_can.h>

#define NODE_ID   0x100   // Node1=0x100, Node2=0x200, Node3=0x300
#define CS_PIN    10

MCP_CAN CAN0(CS_PIN);
byte data[8] = {0, 1, 2, 3, 4, 5, 6, 7};

void setup() {
  Serial.begin(115200);
  while (CAN0.begin(MCP_ANY, CAN_125KBPS, MCP_8MHZ) != CAN_OK) {
    Serial.println("MCP2515 init FAILED - check SPI wiring / power");
    delay(1000);
  }
  Serial.println("MCP2515 init OK");
  CAN0.setMode(MCP_NORMAL);
}

void loop() {
  byte status = CAN0.sendMsgBuf(NODE_ID, 0, 8, data);
  if (status == CAN_OK) {
    Serial.print("Sent ID 0x");
    Serial.print(NODE_ID, HEX);
    Serial.print("  counter=");
    Serial.println(data[0]);
  } else {
    Serial.println("Send FAILED - check bus, termination, other node powered?");
  }
  data[0]++;
  delay(100);
}