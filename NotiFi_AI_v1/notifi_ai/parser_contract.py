"""Constants shared by the ESP CSI parser and deployment preprocessor."""

from __future__ import annotations


CSI_USECOLS = (
    "pc_elapsed_s",
    "sender_id",
    "csi_data",
    "csi_len",
    "first_word_invalid",
    "rssi",
)
LINKS = ("TX1", "TX2", "TX3")
LINK_INDEX = {tx: index for index, tx in enumerate(LINKS)}
N_LINKS = len(LINKS)
CSI_RAW_LEN = 256
N_SUBCARRIERS = CSI_RAW_LEN // 2
GUARD_SUBCARRIERS = (0, 1, 2, 3, 4, 5, 63, 64, 65, 123, 124, 125, 126, 127)
LIVE_SUBCARRIERS = tuple(
    index for index in range(N_SUBCARRIERS) if index not in GUARD_SUBCARRIERS
)
TARGET_FPS = 30.0
CSI_REPRESENTATION = "amp_phase"
MAX_GAP_S = 0.100
PREPROC_VERSION = "v3.0.0"
