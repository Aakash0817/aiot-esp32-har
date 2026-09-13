// Micro-benchmark used to calibrate Wokwi simulated time vs. real ESP32 (see report).
// Result (Wokwi, -Os): 170k float MACs = 167 ms, int32 = 186 ms, int64 = 668 ms, delay(100) = 100.05 ms.
#include <Arduino.h>
void setup(){ Serial.begin(115200); delay(100);
  volatile float fa=1.0001f, fs=0; volatile int32_t ia=3, is=0; volatile int64_t ls=0;
  uint32_t t0=micros(); for(int i=0;i<170000;i++){ fs += fa*fa; } uint32_t tf=micros()-t0;
  t0=micros(); for(int i=0;i<170000;i++){ is += ia*ia; } uint32_t ti=micros()-t0;
  t0=micros(); for(int i=0;i<170000;i++){ ls += (int64_t)ia*ia; ls >>= 1; } uint32_t tl=micros()-t0;
  t0=micros(); delay(100); uint32_t td=micros()-t0;
  Serial.printf("BENCH,float_mac_us=%u,int32_mac_us=%u,int64_us=%u,delay100_us=%u,cpu_mhz=%u\n",tf,ti,tl,td,getCpuFrequencyMhz());
}
void loop(){}
