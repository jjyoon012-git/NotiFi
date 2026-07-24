import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.csi_visualization import save_csi_visualization


def main():
    parser = argparse.ArgumentParser(description="Visualize CSI amplitude from a recorded CSV")
    parser.add_argument("csv", nargs="?", help="Path to CSV file")
    parser.add_argument("--smooth", type=int, default=5, help="Moving average window (default 5)")
    parser.add_argument("--out", type=Path, help="Output PNG path")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        out_path = args.out or csv_path.with_name(f"{csv_path.stem}_csi_visualization.png")
        summary = save_csi_visualization(csv_path, out_path, title=csv_path.stem, smooth_window=args.smooth)
        print(f"Saved: {summary['path']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
