#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_eval_rows(asr_dir: Path, min_eval_id: int) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(asr_dir.glob("*.json"), key=lambda p: int(p.stem)):
        try:
            video_id = int(path.stem)
        except Exception:
            continue
        if video_id < min_eval_id:
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = payload.get("classified_segments", [])
        rows.append(
            {
                "video_id": video_id,
                "video_name": str(payload.get("video_id", "")),
                "video_path": str(payload.get("file_path", "")),
                "fps": float(payload.get("fps", 0.0) or 0.0),
                "total_frames": int(payload.get("total_frames", 0) or 0),
                "language": str(payload.get("language", "")),
                "num_segments": len(segments),
            }
        )
    return rows


def write_manifest_jsonl(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metadata_csv(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_id", "video_name", "video_path", "fps", "total_frames", "language", "num_segments"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BoxComm benchmark manifest files from eval ASR JSONs.")
    parser.add_argument("--asr_dir", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--min_eval_id", type=int, default=478)
    args = parser.parse_args()

    rows = load_eval_rows(Path(args.asr_dir), min_eval_id=int(args.min_eval_id))
    write_manifest_jsonl(rows, Path(args.out_jsonl))
    write_metadata_csv(rows, Path(args.out_csv))

    print(
        json.dumps(
            {
                "video_count": len(rows),
                "min_eval_id": int(args.min_eval_id),
                "out_jsonl": args.out_jsonl,
                "out_csv": args.out_csv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
