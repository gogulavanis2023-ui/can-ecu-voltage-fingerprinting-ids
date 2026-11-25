#include <SPI.h>
#include <mcp_can.h>

#define CS_PIN   PB12
#define LED_PIN  PC13

MCP_CAN CAN0(CS_PIN);

void setup() {
  Serial1.setRx(PA10);
  Serial1.setTx(PA9);
  Serial1.begin(115200);
  delay(200);
  Serial1.println("BOOT");

  SPI.setMOSI(PB15);
  SPI.setMISO(PB14);
  SPI.setSCLK(PB13);
  pinMode(LED_PIN, OUTPUT);

  while (CAN0.begin(MCP_ANY, CAN_125KBPS, MCP_8MHZ) != CAN_OK) {
    Serial1.println("Guardian: init FAILED");
    delay(1000);
  }
  byte m = CAN0.setMode(MCP_NORMAL);
  Serial1.print("setMode result: ");
  Serial1.println(m);   // 0 = OK, anything else = failed
  Serial1.println("Guardian: init OK - listening");
}

void loop() {
  if (CAN0.checkReceive() == CAN_MSGAVAIL) {
    unsigned long rxId = 0;
    byte len = 0;
    byte buf[16];                 // extra room so a bad length can't crash it
    CAN0.readMsgBuf(&rxId, &len, buf);
    if (len > 8) len = 8;

    Serial1.print(millis());
    Serial1.print(" ms  ID 0x");
    Serial1.print(rxId & 0x7FF, HEX);
    Serial1.print("  counter=");
    Serial1.println(buf[0]);

    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
}