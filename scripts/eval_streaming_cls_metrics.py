#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


MODEL_NAME = os.getenv("MODEL_CLASSIFY", os.getenv("MODEL_NAME", "gpt-5.2"))

SYSTEM_MSG = (
    "You are a boxing commentary analyst. "
    "Classify each sentence into exactly one class 1/2/3 and return JSON only."
)

CLASS_DEF = """
CLASS 1 — Play-by-Play Commentary
Immediate, observable ring actions right now.

CLASS 2 — Tactical Commentary
Strategy, intent, momentum, pattern, prediction, or analysis.

CLASS 3 — Contextual Commentary
Background/framing/storyline beyond immediate exchange.
"""


def _get_api_key() -> str:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPENAPI_KEY")
    )
    if not api_key:
        raise EnvironmentError("Missing API key. Set OPENAI_API_KEY (or OPENAI_KEY).")
    return api_key


def build_client(timeout: float = 20.0, max_retries: int = 0) -> OpenAI:
    client_kwargs = {
        "api_key": _get_api_key(),
        "timeout": float(timeout),
        "max_retries": int(max_retries),
    }
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _extract_json(content: str) -> str:
    if content is None:
        return "[]"
    content = str(content).strip()
    if "```json" in content:
        s = content.find("```json") + len("```json")
        e = content.find("```", s)
        if e != -1:
            return content[s:e].strip()
    if "```" in content:
        s = content.find("```") + len("```")
        e = content.find("```", s)
        if e != -1:
            return content[s:e].strip()
    return content


def _safe_chat_completion(client: OpenAI, model: str, messages: List[Dict[str, str]]):
    try:
        return client.chat.completions.create(model=model, messages=messages, temperature=0)
    except Exception as e:
        err = str(e)
        if "temperature" in err and any(k in err for k in ("Unsupported value", "does not support", "unsupported_value")):
            return client.chat.completions.create(model=model, messages=messages)
        raise


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _clean_join(tokens: List[str]) -> str:
    s = " ".join(t.strip() for t in tokens if isinstance(t, str) and t.strip())
    s = re.sub(r"\s+([,\.!\?:;\)\]\}])", r"\1", s)
    s = re.sub(r"([\(\[\{])\s+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_SENT_END_RE = re.compile(r"[\.!\?。！？]+[\"'”’\)\]\}]*$")
_NOISE_RE = re.compile(r"^[\s\.。\,，!！\?？…~\-—_]+$")
_SINGLE_DOT_END_RE = re.compile(r"\.[\"'”’\)\]\}]*$")


def _is_noise_text(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return True
    if t == "...":
        return True
    return bool(_NOISE_RE.fullmatch(t))


def _normalize_stream_text(text: str) -> str:
    t = str(text or "")
    # remove streaming ellipsis noise while keeping normal punctuation
    t = t.replace("…", "...")
    t = re.sub(r"\.\.\.+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # trim dangling punctuation-only wrappers
    t = re.sub(r"^[,，;；:：]+", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_single_word_text(text: str) -> bool:
    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text or ""))
    return len(toks) <= 1


def _cap_sentence_count(rows: List[Dict[str, Any]], max_sentences: int) -> List[Dict[str, Any]]:
    if max_sentences <= 0 or len(rows) <= max_sentences:
        return rows

    data = [dict(x) for x in rows]

    def _merge_at(idx: int) -> None:
        a = data[idx]
        b = data[idx + 1]
        merged = {
            "start_time": float(a["start_time"]),
            "end_time": float(b["end_time"]),
            "duration": round(max(float(b["end_time"]) - float(a["start_time"]), 0.0), 3),
            "text": _clean_join([str(a.get("text", "")), str(b.get("text", ""))]),
        }
        data[idx] = merged
        del data[idx + 1]

    while len(data) > max_sentences and len(data) >= 2:
        best_i = 0
        best_score = float("inf")
        for i in range(len(data) - 1):
            a = data[i]
            b = data[i + 1]
            gap = max(0.0, float(b["start_time"]) - float(a["end_time"]))
            # prefer merging tiny-gap and fragmented pairs first
            short_bonus = -0.2 if (_is_single_word_text(a.get("text", "")) or _is_single_word_text(b.get("text", ""))) else 0.0
            dot_penalty = 0.4 if _SINGLE_DOT_END_RE.search(str(a.get("text", "")).strip()) else 0.0
            score = gap + dot_penalty + short_bonus
            if score < best_score:
                best_score = score
                best_i = i
        _merge_at(best_i)

    return data


def stitch_responses_to_sentences(
    responses: List[Dict[str, Any]],
    sentence_gap_sec: float = 2.0,
    max_sentence_sec: float = 120.0,
    noise_break_sec: float = 4.0,
    target_max_sentences: int = 200,
) -> List[Dict[str, Any]]:
    pieces: List[Dict[str, Any]] = []
    for r in responses:
        tx = _normalize_stream_text(str(r.get("response", "") or "").strip())
        s = float(r.get("start_time", 0.0) or 0.0)
        e = float(r.get("end_time", s) or s)
        if e < s:
            s, e = e, s

        if _is_noise_text(tx):
            pieces.append({"s": s, "e": e, "t": "", "kind": "noise"})
            continue
        pieces.append({"s": s, "e": e, "t": tx, "kind": "speech"})

    pieces.sort(key=lambda x: (x["s"], x["e"]))
    if not pieces:
        return []

    out: List[Dict[str, Any]] = []
    cur_tokens: List[str] = []
    cur_s = 0.0
    cur_e = 0.0

    def _flush_current() -> None:
        nonlocal cur_tokens, cur_s, cur_e
        if not cur_tokens:
            return
        sent = _clean_join(cur_tokens)
        if sent and not _is_noise_text(sent):
            out.append(
                {
                    "start_time": round(cur_s, 3),
                    "end_time": round(cur_e, 3),
                    "duration": round(max(cur_e - cur_s, 0.0), 3),
                    "text": sent,
                }
            )
        cur_tokens = []

    noise_acc = 0.0
    prev_piece_end = None

    for p in pieces:
        ps, pe = float(p["s"]), float(p["e"])
        kind = p.get("kind", "speech")

        if prev_piece_end is not None and ps > prev_piece_end:
            # Natural silence between chunks also counts for break condition
            noise_acc += (ps - prev_piece_end)
        prev_piece_end = pe

        if kind == "noise":
            noise_acc += max(pe - ps, 0.0)
            if noise_acc >= noise_break_sec:
                _flush_current()
            continue

        # speech piece
        gap = (ps - cur_e) if cur_tokens else 0.0
        prev_ended = bool(cur_tokens and _SINGLE_DOT_END_RE.search(cur_tokens[-1]))
        too_long = bool(cur_tokens and (cur_e - cur_s) >= max_sentence_sec)
        if cur_tokens and (gap >= sentence_gap_sec or prev_ended or too_long):
            _flush_current()

        if not cur_tokens:
            cur_s = ps
            cur_e = pe
            cur_tokens = [p["t"]]
        else:
            cur_tokens.append(p["t"])
            cur_e = max(cur_e, pe)

        noise_acc = 0.0

    _flush_current()
    return _cap_sentence_count(out, int(target_max_sentences))


def classify_chunk(
    client: OpenAI,
    model: str,
    chunk_rows: List[Dict[str, Any]],
    pre_ctx: List[Dict[str, Any]],
    post_ctx: List[Dict[str, Any]],
    api_stats: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    ctx_lines: List[str] = []
    if pre_ctx:
        ctx_lines.append("### Context BEFORE (reference only):")
        for s in pre_ctx:
            ctx_lines.append(f'[{s["start_time"]:.2f}-{s["end_time"]:.2f}] "{s["text"]}"')
        ctx_lines.append("")
    if post_ctx:
        ctx_lines.append("### Context AFTER (reference only):")
        for s in post_ctx:
            ctx_lines.append(f'[{s["start_time"]:.2f}-{s["end_time"]:.2f}] "{s["text"]}"')
        ctx_lines.append("")

    target = [{"i": i, "text": r["text"]} for i, r in enumerate(chunk_rows)]

    user_content = (
        f"{CLASS_DEF}\n"
        "Classify ONLY [TARGET] and keep order unchanged.\n"
        "Return JSON ONLY as an array of objects:\n"
        "[{\"i\":0,\"class\":1|2|3}, ...]\n"
        "No explanation.\n\n"
        + "\n".join(ctx_lines)
        + "\n### [TARGET]:\n"
        + json.dumps(target, ensure_ascii=False)
    )

    t0 = time.time()
    resp = _safe_chat_completion(
        client,
        model,
        [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user_content}],
    )
    dt = time.time() - t0
    if api_stats is not None:
        api_stats["calls"] = int(api_stats.get("calls", 0)) + 1
        api_stats.setdefault("durations", []).append(float(dt))
        api_stats.setdefault("batch_sizes", []).append(int(len(chunk_rows)))

    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(_extract_json(raw))
    except Exception:
        parsed = []

    idx2cls: Dict[int, int] = {}
    if isinstance(parsed, list):
        for k, item in enumerate(parsed):
            if isinstance(item, dict):
                idx = item.get("i", k)
                c = item.get("class", 1)
            else:
                idx = k
                c = 1
            try:
                idx = int(idx)
            except Exception:
                idx = k
            try:
                c = int(c)
            except Exception:
                c = 1
            if c not in (1, 2, 3):
                c = 1
            if 0 <= idx < len(chunk_rows):
                idx2cls[idx] = c

    out: List[Dict[str, Any]] = []
    for i, src in enumerate(chunk_rows):
        c = idx2cls.get(i, 1)
        try:
            c = int(c)
        except Exception:
            c = 1
        if c not in (1, 2, 3):
            c = 1
        out.append(
            {
                "start_time": src["start_time"],
                "end_time": src["end_time"],
                "duration": src["duration"],
                "text": src["text"],
                "class": c,
            }
        )
    return out


def classify_sentences(
    client: OpenAI,
    model: str,
    rows: List[Dict[str, Any]],
    gpt_batch_size: int,
    api_stats: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [None] * len(rows)
    need_cls_idx: List[int] = []

    for i, r in enumerate(rows):
        if _is_single_word_text(r.get("text", "")):
            out[i] = {
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "duration": r["duration"],
                "text": r["text"],
                "class": 3,
            }
        else:
            need_cls_idx.append(i)

    gpt_batch_size = max(1, int(gpt_batch_size))
    for st in range(0, len(need_cls_idx), gpt_batch_size):
        idx_batch = need_cls_idx[st: st + gpt_batch_size]
        batch_rows = [rows[i] for i in idx_batch]
        cls_rows = classify_chunk(
            client,
            model,
            batch_rows,
            [],
            [],
            api_stats=api_stats,
        )
        for k, i in enumerate(idx_batch):
            out[i] = cls_rows[k]

    return [x for x in out if x is not None]


def _pctl(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    q = min(100.0, max(0.0, float(q)))
    xs = sorted(vals)
    k = int(round((q / 100.0) * (len(xs) - 1)))
    return float(xs[k])


def _interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    sa, ea = a
    sb, eb = b
    inter = max(0.0, min(ea, eb) - max(sa, sb))
    union = max(ea, eb) - min(sa, sb)
    return (inter / union) if union > 0 else 0.0


def _mean_best_iou(src: List[Tuple[float, float]], tgt: List[Tuple[float, float]]) -> float:
    if not src:
        return 0.0
    if not tgt:
        return 0.0
    vals: List[float] = []
    for it in src:
        vals.append(max(_interval_iou(it, jt) for jt in tgt))
    return sum(vals) / len(vals)


def class_tiou(pred_rows: List[Dict[str, Any]], gt_rows: List[Dict[str, Any]], cls_id: int) -> float:
    pred = [(float(x["start_time"]), float(x["end_time"])) for x in pred_rows if int(x.get("class", -1)) == cls_id]
    gt = [(float(x["start_time"]), float(x["end_time"])) for x in gt_rows if int(x.get("class", -1)) == cls_id]
    p = _mean_best_iou(pred, gt)
    r = _mean_best_iou(gt, pred)
    return 0.5 * (p + r)


def _add_interval_to_minutes(counter: Dict[int, Dict[int, float]], cls_id: int, s: float, e: float) -> None:
    if e <= s:
        return
    m0 = int(s // 60)
    m1 = int((e - 1e-9) // 60)
    for m in range(m0, m1 + 1):
        ms = 60.0 * m
        me = ms + 60.0
        overlap = max(0.0, min(e, me) - max(s, ms))
        if overlap > 0:
            counter[m][cls_id] += overlap


def rows_to_minute_class_durations(rows: List[Dict[str, Any]]) -> Dict[int, Dict[int, float]]:
    out: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        c = int(r.get("class", 1))
        if c not in (1, 2, 3):
            continue
        s = float(r.get("start_time", 0.0) or 0.0)
        e = float(r.get("end_time", s) or s)
        if e < s:
            s, e = e, s
        _add_interval_to_minutes(out, c, s, e)
    return out


def _norm_dist(d: Dict[int, float], eps: float = 1e-8) -> List[float]:
    arr = [float(d.get(1, 0.0)), float(d.get(2, 0.0)), float(d.get(3, 0.0))]
    arr = [x + eps for x in arr]
    z = sum(arr)
    return [x / z for x in arr]


def _kl(p: List[float], q: List[float]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))


def load_gt_rows(asr_dir: str, video_id: int) -> List[Dict[str, Any]]:
    path = os.path.join(asr_dir, f"{video_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    segs = d.get("classified_segments", [])
    out: List[Dict[str, Any]] = []
    for s in segs:
        tx = str(s.get("text", "") or "").strip()
        if not tx:
            continue
        c = int(s.get("class", 1)) if str(s.get("class", "")).strip() else 1
        if c not in (1, 2, 3):
            c = 1
        out.append(
            {
                "start_time": float(s.get("start_time", 0.0) or 0.0),
                "end_time": float(s.get("end_time", 0.0) or 0.0),
                "duration": float(s.get("end_time", 0.0) or 0.0) - float(s.get("start_time", 0.0) or 0.0),
                "text": tx,
                "class": c,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate streaming commentary with class t-IoU and per-minute KL")
    parser.add_argument("--pred_jsonl", type=str, required=True)
    parser.add_argument("--asr_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--api_timeout", type=float, default=20.0)
    parser.add_argument("--api_max_retries", type=int, default=0)
    parser.add_argument("--gpt_batch_size", type=int, default=120)
    parser.add_argument("--context_before", type=int, default=2)
    parser.add_argument("--context_after", type=int, default=1)
    parser.add_argument("--sentence_gap_sec", type=float, default=2.0)
    parser.add_argument("--max_sentence_sec", type=float, default=120.0)
    parser.add_argument("--noise_break_sec", type=float, default=4.0)
    parser.add_argument("--target_max_sentences", type=int, default=200)
    parser.add_argument("--kl_eval_minutes", type=int, default=11)
    parser.add_argument("--min_eval_id", type=int, default=478)
    parser.add_argument("--max_eval_id", type=int, default=999999)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--stitch_only", action="store_true")
    parser.add_argument("--save_pred_sentences_jsonl", type=str, default="")
    parser.add_argument("--save_stitched_sentences_jsonl", type=str, default="")
    args = parser.parse_args()

    rows = _load_jsonl(args.pred_jsonl)
    rows = [r for r in rows if int(r.get("video_id", -1)) >= args.min_eval_id and int(r.get("video_id", -1)) <= args.max_eval_id]
    rows = sorted(rows, key=lambda x: int(x.get("video_id", -1)))
    if int(args.max_videos) > 0:
        rows = rows[: int(args.max_videos)]
    if not rows:
        raise ValueError("No prediction rows in selected id range")

    client = None if args.stitch_only else build_client(timeout=float(args.api_timeout), max_retries=int(args.api_max_retries))

    per_video: List[Dict[str, Any]] = []
    all_cls_scores = {1: [], 2: [], 3: []}
    pred_min_all: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    gt_min_all: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    pred_sent_dump: List[Dict[str, Any]] = []
    stitched_sent_dump: List[Dict[str, Any]] = []
    api_stats: Dict[str, Any] = {"calls": 0, "durations": [], "batch_sizes": []}

    for obj in tqdm(rows, desc="Evaluating videos"):
        vid = int(obj.get("video_id", -1))
        gt = load_gt_rows(args.asr_dir, vid)
        if not gt:
            continue

        pred_sent = stitch_responses_to_sentences(
            obj.get("responses", []),
            sentence_gap_sec=float(args.sentence_gap_sec),
            max_sentence_sec=float(args.max_sentence_sec),
            noise_break_sec=float(args.noise_break_sec),
            target_max_sentences=int(args.target_max_sentences),
        )
        if args.save_stitched_sentences_jsonl:
            stitched_sent_dump.append({"video_id": vid, "stitched_sentences": pred_sent})
        if pred_sent:
            if args.stitch_only:
                pred_cls = []
            else:
                pred_cls = classify_sentences(
                    client,
                    args.model,
                    pred_sent,
                    gpt_batch_size=int(args.gpt_batch_size),
                    api_stats=api_stats,
                )
        else:
            pred_cls = []

        if args.save_pred_sentences_jsonl:
            pred_sent_dump.append({"video_id": vid, "pred_classified_sentences": pred_cls})

        c1 = class_tiou(pred_cls, gt, 1)
        c2 = class_tiou(pred_cls, gt, 2)
        c3 = class_tiou(pred_cls, gt, 3)
        mean3 = (c1 + c2 + c3) / 3.0

        all_cls_scores[1].append(c1)
        all_cls_scores[2].append(c2)
        all_cls_scores[3].append(c3)

        pred_min = rows_to_minute_class_durations(pred_cls)
        gt_min = rows_to_minute_class_durations(gt)
        for m, d in pred_min.items():
            for c, v in d.items():
                pred_min_all[m][c] += v
        for m, d in gt_min.items():
            for c, v in d.items():
                gt_min_all[m][c] += v

        per_video.append(
            {
                "video_id": vid,
                "num_pred_sentences": len(pred_cls),
                "num_gt_sentences": len(gt),
                "t_iou": {
                    "class_1": c1,
                    "class_2": c2,
                    "class_3": c3,
                    "mean_3class": mean3,
                },
            }
        )

    minute_kls: List[Dict[str, Any]] = []
    for m in sorted(set(gt_min_all.keys()) | set(pred_min_all.keys())):
        p = _norm_dist(gt_min_all.get(m, {}))
        q = _norm_dist(pred_min_all.get(m, {}))
        minute_kls.append(
            {
                "minute_index_0based": m,
                "minute_label_1based": m + 1,
                "gt_dist": {"class_1": p[0], "class_2": p[1], "class_3": p[2]},
                "pred_dist": {"class_1": q[0], "class_2": q[1], "class_3": q[2]},
                "kl_gt_pred": _kl(p, q),
            }
        )

    m1 = sum(all_cls_scores[1]) / len(all_cls_scores[1]) if all_cls_scores[1] else 0.0
    m2 = sum(all_cls_scores[2]) / len(all_cls_scores[2]) if all_cls_scores[2] else 0.0
    m3 = sum(all_cls_scores[3]) / len(all_cls_scores[3]) if all_cls_scores[3] else 0.0
    mean_tiou = (m1 + m2 + m3) / 3.0
    kl_eval_minutes = max(1, int(args.kl_eval_minutes))
    minute_kls_eval = [x for x in minute_kls if int(x.get("minute_label_1based", 0)) <= kl_eval_minutes]
    mean_kl = (
        sum(x["kl_gt_pred"] for x in minute_kls_eval) / len(minute_kls_eval)
        if minute_kls_eval
        else 0.0
    )
    api_durations = [float(x) for x in api_stats.get("durations", [])]
    api_batches = [int(x) for x in api_stats.get("batch_sizes", [])]

    result = {
        "pred_jsonl": args.pred_jsonl,
        "asr_dir": args.asr_dir,
        "model": args.model,
        "video_count": len(per_video),
        "eval_id_range": [args.min_eval_id, args.max_eval_id],
        "sentence_reconstruction": {
            "sentence_gap_sec": args.sentence_gap_sec,
            "max_sentence_sec": args.max_sentence_sec,
            "noise_break_sec": args.noise_break_sec,
            "target_max_sentences": args.target_max_sentences,
            "rule": "merge aggressively; split mainly by >2s pause or single-dot end; remove ellipsis noise; cap sentence count",
        },
        "metric_1_sentence_t_iou": {
            "definition": "Per class, symmetric sentence-level t-IoU = 0.5*(mean best IoU pred->gt + mean best IoU gt->pred)",
            "class_1": m1,
            "class_2": m2,
            "class_3": m3,
            "mean_3class": mean_tiou,
        },
        "metric_2_minute_kl": {
            "definition": "Aggregate each minute across all eval videos, build 3-class duration ratio distribution, compute KL(P_gt || P_pred) per minute, then average over first kl_eval_minutes minutes",
            "mean_kl_gt_pred": mean_kl,
            "num_minutes": len(minute_kls_eval),
            "kl_eval_minutes": kl_eval_minutes,
            "per_minute": minute_kls,
        },
        "api_profile": {
            "stitch_only": bool(args.stitch_only),
            "model": args.model,
            "calls": int(api_stats.get("calls", 0)),
            "total_api_seconds": float(sum(api_durations)) if api_durations else 0.0,
            "latency_seconds": {
                "mean": float(statistics.mean(api_durations)) if api_durations else 0.0,
                "median": float(statistics.median(api_durations)) if api_durations else 0.0,
                "p90": _pctl(api_durations, 90),
                "max": max(api_durations) if api_durations else 0.0,
            },
            "gpt_batch_size_stats": {
                "mean": float(statistics.mean(api_batches)) if api_batches else 0.0,
                "median": float(statistics.median(api_batches)) if api_batches else 0.0,
                "min": min(api_batches) if api_batches else 0,
                "max": max(api_batches) if api_batches else 0,
            },
        },
        "per_video": per_video,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.save_pred_sentences_jsonl:
        os.makedirs(os.path.dirname(args.save_pred_sentences_jsonl) or ".", exist_ok=True)
        with open(args.save_pred_sentences_jsonl, "w", encoding="utf-8") as f:
            for r in pred_sent_dump:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.save_stitched_sentences_jsonl:
        os.makedirs(os.path.dirname(args.save_stitched_sentences_jsonl) or ".", exist_ok=True)
        with open(args.save_stitched_sentences_jsonl, "w", encoding="utf-8") as f:
            for r in stitched_sent_dump:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps({
        "video_count": result["video_count"],
        "metric_1_mean_3class_t_iou": result["metric_1_sentence_t_iou"]["mean_3class"],
        "metric_1_class_1": result["metric_1_sentence_t_iou"]["class_1"],
        "metric_1_class_2": result["metric_1_sentence_t_iou"]["class_2"],
        "metric_1_class_3": result["metric_1_sentence_t_iou"]["class_3"],
        "metric_2_mean_minute_kl": result["metric_2_minute_kl"]["mean_kl_gt_pred"],
        "metric_2_num_minutes": result["metric_2_minute_kl"]["num_minutes"],
        "api_calls": result["api_profile"]["calls"],
        "api_latency_mean_s": result["api_profile"]["latency_seconds"]["mean"],
        "api_latency_p90_s": result["api_profile"]["latency_seconds"]["p90"],
        "output_json": args.output_json,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
