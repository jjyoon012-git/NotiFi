from __future__ import annotations

import csv
from dataclasses import dataclass


EXPECTED_TX = {
    "1a:00:00:00:00:00": "TX1",
    "1a:00:00:00:00:01": "TX2",
    "1a:00:00:00:00:02": "TX3",
}


@dataclass(frozen=True)
class CsiFrame:
    sequence_number: str
    sender_mac: str
    sender_id: str
    rssi: str
    rate: str
    noise_floor: str
    fft_gain: str
    agc_gain: str
    channel: str
    firmware_timestamp_us: str
    sig_len: str
    rx_state: str
    csi_len: str
    first_word_invalid: str
    csi_data: str
    raw_line: str


def parse_csi_line(line: str) -> CsiFrame | None:
    line = line.strip()
    if not line.startswith("CSI_DATA,"):
        return None
    try:
        parts = next(csv.reader([line]))
    except (csv.Error, StopIteration):
        return None
    if len(parts) < 15:
        return None
    sender_mac = parts[2].strip().lower()
    return CsiFrame(
        sequence_number=parts[1].strip(),
        sender_mac=sender_mac,
        sender_id=EXPECTED_TX.get(sender_mac, "UNKNOWN"),
        rssi=parts[3].strip(),
        rate=parts[4].strip(),
        noise_floor=parts[5].strip(),
        fft_gain=parts[6].strip(),
        agc_gain=parts[7].strip(),
        channel=parts[8].strip(),
        firmware_timestamp_us=parts[9].strip(),
        sig_len=parts[10].strip(),
        rx_state=parts[11].strip(),
        csi_len=parts[12].strip(),
        first_word_invalid=parts[13].strip(),
        csi_data=parts[14].strip(),
        raw_line=line,
    )
