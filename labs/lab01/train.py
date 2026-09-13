from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train import parse_args, run


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir == Path("outputs"):
        args.output_dir = Path(__file__).resolve().parent / "outputs"
    run(args.input, args.output_dir, args.random_state)
