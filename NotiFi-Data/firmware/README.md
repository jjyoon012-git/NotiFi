# 3TX + 1RX ESP32-C6 펌웨어 가이드

대상은 XIAO ESP32-C6 네 대(TX 3대, RX 1대)다. ESP-IDF target은 `esp32c6`이며 ESP-CSI `examples/get-started/csi_send`, `csi_recv`를 사용한다.

## RX 변경

`esp-csi/examples/get-started/csi_recv/main/app_main.c`

헤더:

```c
#include <stdbool.h>
```

허용 MAC:

```c
static const uint8_t CONFIG_CSI_ALLOWED_MACS[][6] = {
    {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x1a, 0x00, 0x00, 0x00, 0x00, 0x01},
    {0x1a, 0x00, 0x00, 0x00, 0x00, 0x02},
};
#define CONFIG_CSI_ALLOWED_MAC_NUM 3
```

`wifi_csi_rx_cb`의 단일 MAC 필터를 교체한다.

```c
bool matched = false;
for (int i = 0; i < CONFIG_CSI_ALLOWED_MAC_NUM; i++) {
    if (memcmp(info->mac, CONFIG_CSI_ALLOWED_MACS[i], 6) == 0) {
        matched = true;
        break;
    }
}
if (!matched) {
    return;
}
```

RX 자신의 `esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC)` 줄은 변경하지 않는다.

## TX 변경

`esp-csi/examples/get-started/csi_send/main/app_main.c`

세 TX 모두:

```c
#define CONFIG_SEND_FREQUENCY 30
```

각 보드 MAC:

```c
// TX1
static const uint8_t CONFIG_CSI_SEND_MAC[] =
    {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

// TX2
static const uint8_t CONFIG_CSI_SEND_MAC[] =
    {0x1a, 0x00, 0x00, 0x00, 0x00, 0x01};

// TX3
static const uint8_t CONFIG_CSI_SEND_MAC[] =
    {0x1a, 0x00, 0x00, 0x00, 0x00, 0x02};
```

속도가 바뀌므로 TX1도 반드시 다시 플래시한다. MAC을 바꿀 때마다 다시 build한다.

## XIAO ESP32-C6 외장 안테나

TX/RX 모두 Wi-Fi 초기화 전에 실행한다.

```c
gpio_set_direction(GPIO_NUM_3, GPIO_MODE_OUTPUT);
gpio_set_level(GPIO_NUM_3, 0);
vTaskDelay(pdMS_TO_TICKS(100));
gpio_set_direction(GPIO_NUM_14, GPIO_MODE_OUTPUT);
gpio_set_level(GPIO_NUM_14, 1);
```

## 채널

TX 3대와 RX 1대의 `CONFIG_LESS_INTERFERENCE_CHANNEL`을 모두 같은 값 `11`로 고정한다. dataset version 도중 채널이나 packet rate를 바꾸지 않는다.

## Windows 플래시

ESP-IDF Command Prompt에서 실행한다.

```bat
cd C:\PATH\TO\esp-idf
export.bat
```

RX:

```bat
cd C:\PATH\TO\esp-csi\examples\get-started\csi_recv
idf.py set-target esp32c6
idf.py -p COM_RX build flash monitor
```

`Ctrl+]`로 monitor를 종료한다.

TX3의 MAC을 `02`로 설정하고:

```bat
cd C:\PATH\TO\esp-csi\examples\get-started\csi_send
idf.py set-target esp32c6
idf.py -p COM_TX3 build flash
```

TX2의 MAC을 `01`로 바꾸고 다시 build/flash한다.

```bat
idf.py -p COM_TX2 build flash
```

TX1의 MAC을 `00`으로 바꾸고 다시 build/flash한다.

```bat
idf.py -p COM_TX1 build flash
```

최종 확인:

```powershell
python scripts/check_tx_links.py COM_RX --rate 30
```

TX1/TX2/TX3가 10초 동안 각각 1 frame 이상이면 본 수집을 시작할 수 있다. 240/270 frames 같은 낮은 수신량 기준은 경고나 중단 조건으로 쓰지 않는다.
