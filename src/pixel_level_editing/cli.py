from __future__ import annotations

import argparse
from pathlib import Path

from .core import load_rounds_config, process_pair, process_sequence, save_pair_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-edit",
        description="Restore untouched pixels while keeping intentional image edits.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pair = commands.add_parser("pair", help="Process one before/after image pair")
    pair.add_argument("--before", required=True, help="Baseline image")
    pair.add_argument("--after", required=True, help="Edited image")
    pair.add_argument("--output", required=True, help="Output directory")
    pair.add_argument("--mask", help="Optional binary mask, including SAM 3.1 output")
    pair.add_argument("--mode", choices=("paste", "recolor"), default="paste")
    pair.add_argument("--threshold", type=float, default=24.0)
    pair.add_argument("--min-area-ratio", type=float, default=0.0005)
    pair.add_argument("--dilate", type=int, default=5)

    sequence = commands.add_parser("sequence", help="Process a cumulative multi-round sequence")
    sequence.add_argument("--config", required=True, help="JSON workflow config")
    sequence.add_argument("--original", help="Override the original image from config")
    sequence.add_argument("--output", required=True, help="Output directory")

    app = commands.add_parser("app", help="Launch the local web interface")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=7860)
    app.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pair":
        result = process_pair(
            args.before,
            args.after,
            mask_path=args.mask,
            mode=args.mode,
            threshold=args.threshold,
            min_area_ratio=args.min_area_ratio,
            dilate=args.dilate,
        )
        paths = save_pair_result(result, args.output)
        print(paths["result"])
        return

    if args.command == "sequence":
        config_original, rounds = load_rounds_config(args.config)
        original = args.original or config_original
        if not original:
            raise SystemExit("Provide --original or set 'original' in the JSON config")
        process_sequence(original, rounds, args.output)
        print(str(Path(args.output) / "manifest.json"))
        return

    if args.command == "app":
        from .app import launch

        launch(args.host, args.port, args.share)


if __name__ == "__main__":
    main()

