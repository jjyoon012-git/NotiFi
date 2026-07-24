# 3TX + 1RX Flash Guide

이 문서는 `README.md`의 상세 플래시 절차를 따릅니다.

핵심만 요약하면 다음 순서입니다.

1. `firmware/csi_recv/main/app_main.c`를 esp-csi의 `csi_recv/main/app_main.c`에 복사
2. `firmware/csi_send/main/app_main.c`를 esp-csi의 `csi_send/main/app_main.c`에 복사
3. RX 보드 1개 플래시
4. `set_tx_mac.py --tx tx3` 후 TX3 플래시
5. `set_tx_mac.py --tx tx2` 후 TX2 플래시
6. `set_tx_mac.py --tx tx1` 후 TX1 플래시
7. RX만 노트북에 연결하고 TX 3개 전원 켠 뒤 `check_tx_links.py` 실행

성공 판단은 실제로 `CSI_DATA`가 들어오고, MAC별 카운트가 고르게 나오는 경우에만 합니다.
