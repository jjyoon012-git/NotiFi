import csv

from notifi_collection.csi_visualization import load_amplitude_series, save_csi_visualization


def test_save_csi_visualization_creates_png(tmp_path) -> None:
    csv_path = tmp_path / "sample_csi.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pc_elapsed_s", "sender_id", "csi_data"),
        )
        writer.writeheader()
        for sender in ("TX1", "TX2", "TX3"):
            for index in range(3):
                writer.writerow(
                    {
                        "pc_elapsed_s": f"{index * 0.1:.3f}",
                        "sender_id": sender,
                        "csi_data": "[1,2,3,4,5,6]",
                    }
                )

    series = load_amplitude_series(csv_path)
    assert {sender: len(values[0]) for sender, values in series.items()} == {
        "TX1": 3,
        "TX2": 3,
        "TX3": 3,
    }

    out_path = tmp_path / "sample_csi_visualization.png"
    summary = save_csi_visualization(csv_path, out_path)
    assert out_path.exists()
    assert summary["tx1_frames"] == 3
    assert summary["tx2_frames"] == 3
    assert summary["tx3_frames"] == 3
