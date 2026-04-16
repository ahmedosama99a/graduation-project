#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_WS   25
#define I2S_SD   33
#define I2S_SCK  26
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE     16000
#define RECORD_SECONDS  10
#define BUFFER_SAMPLES  512

const uint8_t MAGIC[8] = {'L','U','N','G','W','A','V','1'};

int32_t i2sBuffer[BUFFER_SAMPLES];
int16_t pcmBuffer[BUFFER_SAMPLES];

void setupI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  i2s_zero_dma_buffer(I2S_PORT);
}

void convert32to16(int16_t *dest, int32_t *src, int count) {
  for (int i = 0; i < count; i++) {
    int32_t s = src[i] >> 14;
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    dest[i] = (int16_t)s;
  }
}

void recordAndSendOnce() {
  uint32_t totalSamples = SAMPLE_RATE * RECORD_SECONDS;
  uint32_t totalBytes = totalSamples * sizeof(int16_t);
  uint32_t sentSamples = 0;

  Serial.write(MAGIC, 8);
  Serial.write((uint8_t *)&totalBytes, 4);

  while (sentSamples < totalSamples) {
    size_t bytesRead = 0;

    i2s_read(I2S_PORT, (void *)i2sBuffer, sizeof(i2sBuffer), &bytesRead, portMAX_DELAY);

    int samplesRead = bytesRead / sizeof(int32_t);
    if (samplesRead <= 0) continue;

    if (sentSamples + samplesRead > totalSamples) {
      samplesRead = totalSamples - sentSamples;
    }

    convert32to16(pcmBuffer, i2sBuffer, samplesRead);
    Serial.write((uint8_t *)pcmBuffer, samplesRead * sizeof(int16_t));

    sentSamples += samplesRead;
  }

  Serial.flush();
}

void setup() {
  Serial.begin(921600);
  delay(1000);
  setupI2S();
  delay(300);
  recordAndSendOnce();
}

void loop() {
}
