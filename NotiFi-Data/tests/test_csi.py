from notifi_collection.csi import parse_csi_line


def test_parse_csi_line() -> None:
    line = (
        'CSI_DATA,1664386,1a:00:00:00:00:02,-69,11,-99,16,55,11,'
        '59354667,47,2,256,0,"[0,1,-2,3]"'
    )
    frame = parse_csi_line(line)
    assert frame is not None
    assert frame.sender_id == "TX3"
    assert frame.sequence_number == "1664386"
    assert frame.firmware_timestamp_us == "59354667"
    assert frame.csi_data == "[0,1,-2,3]"


def test_ignore_non_csi_and_short_rows() -> None:
    assert parse_csi_line("I (100) csi_recv: hello") is None
    assert parse_csi_line("CSI_DATA,1,2") is None

