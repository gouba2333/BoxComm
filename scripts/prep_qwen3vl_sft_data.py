#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

EVAL_START_ID = 478


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def resolve_video_path(video_dir: str, video_id: int) -> Optional[str]:
    exact_candidates = [
        os.path.join(video_dir, f"{video_id}.mp4"),
        os.path.join(video_dir, f"{video_id:04d}.mp4"),
        os.path.join(video_dir, f"{video_id:05d}.mp4"),
    ]
    for p in exact_candidates:
        if os.path.exists(p):
            return p

    prefix_matches = sorted(glob.glob(os.path.join(video_dir, f"{video_id}*.mp4")))
    if prefix_matches:
        return prefix_matches[0]

    fuzzy_matches = sorted(glob.glob(os.path.join(video_dir, f"*{video_id}*.mp4")))
    if fuzzy_matches:
        return fuzzy_matches[0]

    return None


def resolve_event_json_path(event_dir: str, video_path: str) -> Optional[str]:
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    p = os.path.join(event_dir, video_stem, "video_event_inference_3.json")
    return p if os.path.exists(p) else None


def _extract_fps(payload: Dict[str, Any], default: float = 30.0) -> float:
    candidate_keys = ["fps", "video_fps", "avg_fps", "frame_rate"]
    for k in candidate_keys:
        if k in payload:
            fps = _safe_float(payload.get(k), default)
            if fps > 0:
                return fps
    return default


def _extract_event_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    preferred_keys = [
        "events",
        "event_list",
        "video_events",
        "punch_events",
        "results",
        "predictions",
        "data",
    ]
    for k in preferred_keys:
        v = payload.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]

    out: List[Dict[str, Any]] = []

    def _dfs(node: Any):
        if isinstance(node, dict):
            lk = {str(k).lower() for k in node.keys()}
            hint_keys = {"start_frame", "starttime", "start_time", "side", "technique", "target", "effect", "result"}
            if lk & hint_keys:
                out.append(node)
            for v in node.values():
                _dfs(v)
        elif isinstance(node, list):
            for item in node:
                _dfs(item)

    _dfs(payload)
    return out


def _parse_side(side_raw: Any) -> str:
    s = str(side_raw or "").strip().lower()
    if s in {"red", "blue"}:
        return s
    if "red" in s:
        return "red"
    if "blue" in s:
        return "blue"
    return s if s else "unknown"


def _parse_technique(technique_raw: Any) -> str:
    t = str(technique_raw or "").strip().lower()
    if not t:
        return "unknown punch"

    hand = ""
    for ch in t:
        if ch in {"l", "r"}:
            hand = "left" if ch == "l" else "right"
            break

    punch_type = ""
    if "straight" in t:
        punch_type = "straight"
    elif "hook" in t:
        punch_type = "hook"
    elif "uppercut" in t:
        punch_type = "uppercut"

    if hand and punch_type:
        return f"{hand} {punch_type}"
    if punch_type:
        return punch_type
    return t.replace("_", " ").replace("-", " ")


def _parse_target(target_raw: Any) -> str:
    t = str(target_raw or "").strip().lower()
    if t in {"chest", "abdomen", "torso"}:
        return "torso"
    if "chest" in t or "abdomen" in t or "torso" in t:
        return "torso"
    if "head" in t:
        return "head"
    return ""


def _parse_event_time_seconds(ev: Dict[str, Any], fps_default: float = 30.0) -> Optional[float]:
    if "start_time" in ev:
        return _safe_float(ev.get("start_time"), None)
    if "startTime" in ev:
        return _safe_float(ev.get("startTime"), None)

    frame_keys = [
        "frame_begin",
        "frameBegin",
        "start_frame",
        "startFrame",
        "frame_start",
        "start",
        "start_idx",
    ]
    frame = None
    for k in frame_keys:
        if k in ev:
            frame = _safe_float(ev.get(k), None)
            break
    if frame is None:
        return None

    fps = _safe_float(ev.get("fps", fps_default), fps_default)
    if fps <= 0:
        fps = fps_default
    return frame / fps


def _event_to_text(ev: Dict[str, Any], fps_default: float = 30.0) -> Optional[Tuple[float, str]]:
    t = _parse_event_time_seconds(ev, fps_default=fps_default)
    if t is None:
        return None

    side = _parse_side(ev.get("side", ev.get("boxer_side", ev.get("player_side"))))
    technique = _parse_technique(ev.get("technique", ev.get("punch_type", ev.get("type"))))

    effect = str(ev.get("effect", ev.get("result", ev.get("outcome", ""))) or "").strip().lower()
    target = _parse_target(ev.get("target", ev.get("hit_target", ev.get("part", ev.get("region", "")))))

    landing_desc = ""
    if effect in {"effective", "touch"} and target in {"head", "torso"}:
        landing_desc = f", land on {target}"

    text = f"[{t:.1f}s] {side}, {technique}{landing_desc}."
    return t, text


def load_video_events(event_json_path: Optional[str], fps_default: float = 30.0) -> List[Tuple[float, str]]:
    if not event_json_path or not os.path.exists(event_json_path):
        return []

    try:
        with open(event_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    raw_events = _extract_event_list(payload)
    parsed: List[Tuple[float, str]] = []
    for ev in raw_events:
        row = _event_to_text(ev, fps_default=fps_default)
        if row is None:
            continue
        parsed.append(row)

    parsed.sort(key=lambda x: x[0])
    return parsed


def build_previous_events(events: List[Tuple[float, str]], t_mid: float, max_history: int = 32) -> List[str]:
    if not events:
        return []

    past = [txt for t, txt in events if t < t_mid]
    if len(past) <= max_history:
        return past
    return past[-max_history:]


def build_previous_text(segments: List[Dict], cur_idx: int, max_history: int = 8) -> str:
    left = max(0, cur_idx - max_history)
    hist = []
    for s in segments[left:cur_idx]:
        t = _safe_float(s.get("start_time", 0.0), 0.0)
        txt = str(s.get("text", "")).strip()
        if not txt:
            continue
        hist.append(f"[{t:.1f}s] {txt}")
    return " ".join(hist).strip()


def resolve_split_config(args: argparse.Namespace) -> Tuple[str, int, Optional[int], str]:
    split_name = "custom"
    min_id = int(args.min_id)
    max_id = args.max_id
    output_jsonl = str(args.output_jsonl).strip()

    if args.train:
        split_name = "train"
        min_id = 0
        max_id = EVAL_START_ID
        if not output_jsonl:
            output_jsonl = "sft_qwen3vl_train.jsonl"
    elif args.eval:
        split_name = "eval"
        min_id = EVAL_START_ID
        max_id = None
        if not output_jsonl:
            output_jsonl = "sft_qwen3vl_eval.jsonl"
    else:
        if not output_jsonl:
            output_jsonl = "sft_qwen3vl.jsonl"

    return split_name, min_id, max_id, output_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="/hdd/wangkw/dataset/shijs_processed/")
    parser.add_argument("--json_dir", type=str, default="/mnt/nvme_share/wangkw/ASR/output_whisperx_labeled_realign_timing_fixed_v3/")
    parser.add_argument("--event_dir", type=str, default="/hdd/wangkw/dataset/shijs_processed_event/")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--train", action="store_true", help=f"Prepare training split with video id < {EVAL_START_ID}.")
    split_group.add_argument("--eval", action="store_true", help=f"Prepare evaluation split with video id >= {EVAL_START_ID}.")
    parser.add_argument("--min_id", type=int, default=0)
    parser.add_argument("--max_id", type=int, default=EVAL_START_ID)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--strict_video_exists", action="store_true", default=False)
    parser.add_argument("--event_history", type=int, default=32)
    parser.add_argument("--event_fps_default", type=float, default=30.0)
    args = parser.parse_args()

    split_name, min_id, max_id, output_jsonl = resolve_split_config(args)
    json_files = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    total_json = 0
    total_segments = 0
    kept_segments = 0
    skipped_no_video = 0
    skipped_bad_json = 0
    missing_event_file = 0
    event_cache: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for jf in json_files:
            stem = os.path.splitext(os.path.basename(jf))[0]
            vid = _safe_int(stem, -1)
            if vid < min_id:
                continue
            if max_id is not None and vid >= max_id:
                continue

            total_json += 1

            try:
                with open(jf, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                skipped_bad_json += 1
                continue

            video_fps = _extract_fps(payload, default=args.event_fps_default)

            segments = payload.get("classified_segments", [])
            if not isinstance(segments, list) or len(segments) == 0:
                continue

            video_path = resolve_video_path(args.video_dir, vid)
            if video_path is None and args.strict_video_exists:
                skipped_no_video += len(segments)
                continue

            effective_video_path = video_path if video_path is not None else os.path.join(args.video_dir, f"{vid}.mp4")
            cache_key = (vid, int(round(video_fps * 1000)))
            if cache_key not in event_cache:
                event_json_path = resolve_event_json_path(args.event_dir, effective_video_path)
                if event_json_path is None:
                    missing_event_file += 1
                event_cache[cache_key] = load_video_events(event_json_path, fps_default=video_fps)
            parsed_events = event_cache.get(cache_key, [])

            for i, seg in enumerate(segments):
                total_segments += 1
                target_text = str(seg.get("text", "")).strip()
                if not target_text:
                    continue

                t_mid = round(_safe_float(seg.get("start_time", 0.0), 0.0), 2)
                target_class = _safe_int(seg.get("class", -1), -1)
                previous_text = build_previous_text(segments, i, max_history=8)
                previous_events = build_previous_events(parsed_events, t_mid=t_mid, max_history=args.event_history)

                row = {
                    "video_id": vid,
                    "video_path": effective_video_path,
                    "t_mid": t_mid,
                    "target_text": target_text,
                    "target_class": target_class,
                    "previous_text": previous_text,
                    "previous_events": previous_events,
                    "segment_index": i,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_segments += 1

    print(
        json.dumps(
            {
                "split": split_name,
                "id_range": {
                    "min_id": min_id,
                    "max_id_exclusive": max_id,
                },
                "output": output_jsonl,
                "processed_json_files": total_json,
                "total_segments": total_segments,
                "kept_segments": kept_segments,
                "skipped_bad_json": skipped_bad_json,
                "skipped_no_video_segments": skipped_no_video,
                "videos_missing_event_file": missing_event_file,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
