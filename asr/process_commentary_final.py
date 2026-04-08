"""
process_commentary_final.py
───────────────────────────
Self-contained final pipeline from WhisperX ASR to labeled commentary JSON.

Flow:
1) Load raw WhisperX segments/words from output_whisperx.
2) GPT boxing-term correction + sentence split.
3) Unified boundary/timing post-process: pause split + global realignment + timing fix.
4) LLM-assisted middle-gap split (pause-aware, multi-candidate).
5) Final class (1/2/3) classification with context window.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import threading
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

# ───────────────────────── Configuration ─────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")
SPLIT_MODEL = os.getenv("MODEL_SPLIT", MODEL_NAME)
CLASSIFY_MODEL = os.getenv("MODEL_CLASSIFY", MODEL_NAME)

INPUT_DIR = "./output_whisperx"
OUTPUT_DIR = "./output_whisperx_labeled_final"

CLASSIFY_BATCH_SIZE = int(os.getenv("CLASSIFY_BATCH_SIZE", "20"))
CONTEXT_BEFORE = int(os.getenv("CONTEXT_BEFORE", "5"))
CONTEXT_AFTER = int(os.getenv("CONTEXT_AFTER", "3"))

PAUSE_SPLIT_THRESHOLD = float(os.getenv("PAUSE_SPLIT_THRESHOLD", "0.0"))
PAUSE_SPLIT_MIN_WORDS = int(os.getenv("PAUSE_SPLIT_MIN_WORDS", "2"))

MAX_WORD_DURATION = float(os.getenv("MAX_WORD_DURATION", "1.0"))
LONG_GAP_SEC = float(os.getenv("LONG_GAP_SEC", "2.0"))
MOVE_TOL_SEC = float(os.getenv("MOVE_TOL_SEC", "1.0"))

ENABLE_MID_GAP_SPLIT = os.getenv("ENABLE_MID_GAP_SPLIT", "1") == "1"
MID_GAP_THRESHOLD = float(os.getenv("MID_GAP_THRESHOLD", "2.0"))
MID_GAP_MAX_CANDIDATES = int(os.getenv("MID_GAP_MAX_CANDIDATES", "3"))

LOG_FILE = "./log_final.txt"
LOG_LOCK = threading.Lock()


# ───────────────────────── Utilities ─────────────────────────
def write_log(message: str, is_error: bool = False) -> None:
    with LOG_LOCK:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p = "[ERROR]" if is_error else "[INFO]"
                f.write(f"{ts} {p} {message}\n")
        except Exception:
            pass


def _get_api_key() -> str:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPENAPI_KEY")
    )
    if not api_key:
        raise EnvironmentError("Missing API key. Set OPENAI_API_KEY (or OPENAI_KEY).")
    return api_key


def build_client() -> OpenAI:
    client_kwargs = {
        "api_key": _get_api_key(),
        "timeout": 60.0,
        "max_retries": 2,
    }
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def extract_json(content: str) -> str:
    if content is None:
        return "[]"
    c = str(content).strip()
    if "```json" in c:
        s = c.find("```json") + len("```json")
        e = c.find("```", s)
        if e != -1:
            return c[s:e].strip()
    elif "```" in c:
        s = c.find("```") + len("```")
        e = c.find("```", s)
        if e != -1:
            return c[s:e].strip()
    return c if c else "[]"


def _safe_chat_completion(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: Any = None):
    try:
        params: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        resp = client.chat.completions.create(**params)
        if not resp or not resp.choices:
            raise ValueError(f"Empty response from model '{model}'")
        return resp
    except Exception as e:
        err = str(e)
        if "temperature" in err and any(k in err for k in ("Unsupported value", "does not support", "unsupported_value")):
            resp = client.chat.completions.create(model=model, messages=messages)
            if not resp or not resp.choices:
                raise ValueError(f"Empty response from model '{model}' (no-temp retry)")
            return resp
        raise


# ───────────────────────── Prompts ─────────────────────────
CLASS_SYSTEM = (
    "You are a boxing commentary analyst. "
    "Classify each sentence into exactly one class based on what it conveys and whether it can be seen on screen. "
    "Use surrounding context only to disambiguate — classify ONLY the explicitly marked sentences. "
    "Return ONLY a valid JSON array, no markdown, no explanation."
)

CLASS_DEF = """\
CLASS 1 — Play-by-Play Commentary
A specific, observable action or ring event happening RIGHT NOW, verifiable from a 1–3 s clip.

Qualifies as Class 1:
• Named punch / exchange / defence / movement:
  "jab lands", "steps back", "slips the jab", "one-two combination", "backhand to the body again"
• Current position / state: "back on the ropes", "clinch", "heads clash", "off balance"
• Ring events: "referee warns", "bell", "standing count", "score 10-9"
• Direct exclamation reacting to a specific visible action: "Oh! Great shot!", "Brilliant!", "Lovely right hand!"
• "trying to" / "targeting" + named physical action or body part:
    "targeting the body", "trying to land the hook", "going to the body" → Class 1
• Demonstrative + specific named action: "Look at that hook!", "That right hand just got through"

Does NOT qualify as Class 1 — use Class 2 instead:
• "trying to" + abstract goal: "trying to find his range", "trying to force the pace"
• Comparative / quality judgment: "that's a bit better", "a bright start", "growing into the round"
• Habitual pattern ("feinting at times", "has been X-ing") — pattern over multiple moments, not one action

CLASS 2 — Tactical Commentary
Requires inference, pattern recognition, or professional boxing knowledge.

• Intent / tactics: "trying to find his range", "looking to dictate", "playing for the counter"
• Comparative judgments: "that's a bit better", "starting to find his range"
• Pattern / trend / momentum: "controlling the distance", "has been the busier fighter",
  "tiring as the rounds go on"
• General quality (no named trigger): "a bright start", "impressive display", "growing into the round"
• Prescriptions / predictions: "needs to cut the ring off", "should go to the body"
• Summary / consequence: "he's won that exchange", "second time X has been warned"

CLASS 3 — Contextual Commentary
Information that frames and contextualizes the action beyond the immediate live exchange.
These lines provide narrative context for understanding fighters, event setting, and storyline.

• Fighter bios / career / physique: name, nationality, record, titles ("gold medal", "world champion"),
  physical build / strength
• Event info: venue, weight division, championship name, round structure
• History / culture: past results, previous meetings, career arc; generic nationality style claims
• Cross-match references (past tense OR present perfect): "I saw a knockdown too",
    "I have seen him put someone to sleep", "in their previous meeting" → Class 3
• Broadcast filler: commentary intros, crowd atmosphere, score-card disputes
  ("I strongly disagree with this"), timing calls ("we've had 30 seconds"),
  transitions ("here we go", "underway"), replay intros ("let's look at the action from round 1"),
  short vague affirmations / cross-talk ("Yeah.","Absolutely.", "Exactly.")
"""

SPLIT_SYSTEM = (
    "You are a text segmentation tool for boxing commentary. "
    "Split ASR chunks into self-contained sentences using ACTION-ANALYSIS split and related rules. "
    "Do NOT paraphrase or rewrite content; preserve original wording as much as possible. "
    "Also silently correct obvious boxing ASR homophones (e.g. job->jab, fainting->feinting). "
    "Return ONLY a JSON array of strings."
)

MID_SPLIT_SYSTEM = (
    "You are assisting sentence-boundary validation for boxing commentary ASR. "
    "Given ONE candidate middle-pause boundary, decide split or not. "
    "Pause is a strong cue: >=4.0s usually split unless grammar tightly bound; "
    "2.0-4.0s split if right side starts a new proposition/discourse move. "
    "Do not split fixed collocations. "
    "Return JSON only with keys: split, reason, left_class, right_class."
)

# ───────────────────────── Text/word helpers ─────────────────────────
def _normalize_token(s: str) -> str:
    return re.sub(r"[^\w']+", "", (s or "").lower())


def _sent_tokens(text: str) -> List[str]:
    return [t for t in (_normalize_token(x) for x in (text or "").split()) if t]


def _sim(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(a=a, b=b).ratio()
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    f1 = (2 * inter) / (len(a) + len(b))
    return 0.6 * seq + 0.4 * f1


def _words_to_text(words: List[Dict[str, Any]]) -> str:
    out = ""
    punct_no_space = {",", ".", "!", "?", ":", ";", "%", ")", "]", "}"}
    contractions = {"'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}
    for w in words:
        tok = str(w.get("word", "")).strip()
        if not tok:
            continue
        if not out:
            out = tok
        elif tok in punct_no_space or tok in contractions:
            out += tok
        elif out.endswith("(") or out.endswith("[") or out.endswith("{"):
            out += tok
        else:
            out += " " + tok
    return out


def _refresh_segment(seg: Dict[str, Any], fps: float) -> None:
    words = seg.get("words", []) or []
    if not words:
        seg["text"] = seg.get("text", "")
        return
    words.sort(key=lambda w: float(w.get("start_time", 0.0) or 0.0))
    st = float(words[0].get("start_time", seg.get("start_time", 0.0)) or 0.0)
    et = float(words[-1].get("end_time", seg.get("end_time", st)) or st)
    seg["start_time"] = round(st, 3)
    seg["end_time"] = round(et, 3)
    seg["start_frame"] = int(round(st * fps))
    seg["end_frame"] = int(round(et * fps))
    seg["text"] = _words_to_text(words)


def _parse_fighter_names(video_id: str) -> Tuple[str, str]:
    parts = (video_id or "").split("_")
    if len(parts) >= 3:
        blue = " ".join(parts[1].split("-"))
        red = " ".join("_".join(parts[2:]).split("-"))
        return blue.strip(), red.strip()
    return "", ""


# ───────────────────────── Semantic split ─────────────────────────
def _assign_words_to_parts(words: List[Dict[str, Any]], parts: List[str], pause_snap_sec: float = 0.6, snap_window: int = 4):
    result = [[] for _ in parts]
    if not words or not parts:
        return result
    if len(parts) == 1:
        result[0] = words
        return result

    part_counts = [sum(1 for t in p.split() if _normalize_token(t)) for p in parts]
    token_word_idx = [i for i, w in enumerate(words) if _normalize_token(w.get("word", ""))]
    if not token_word_idx:
        result[0] = words
        return result

    splits = []
    cum = 0
    prev = -1
    for i in range(len(parts) - 1):
        cum += part_counts[i]
        t_idx = max(0, min(cum - 1, len(token_word_idx) - 2))
        w_idx = token_word_idx[t_idx]
        lo = max(prev + 1, w_idx - snap_window)
        hi = min(len(words) - 2, w_idx + snap_window)
        best_k, best_gap = w_idx, -1.0
        for k in range(lo, hi + 1):
            g = float(words[k + 1].get("start_time", 0.0) or 0.0) - float(words[k].get("end_time", 0.0) or 0.0)
            if g > best_gap:
                best_gap, best_k = g, k
        split_idx = best_k if best_gap >= pause_snap_sec else w_idx
        min_allowed = prev + 1
        max_allowed = len(words) - (len(parts) - (i + 1)) - 1
        split_idx = max(min_allowed, min(split_idx, max_allowed))
        splits.append(split_idx)
        prev = split_idx

    start = 0
    for i, e in enumerate(splits):
        result[i] = words[start:e + 1]
        start = e + 1
    result[-1] = words[start:]
    return result


def _split_one_segment(client: OpenAI, seg: Dict[str, Any], fps: float, fighter_names: Tuple[str, str]) -> List[Dict[str, Any]]:
    text = (seg.get("text", "") or "").strip()
    words = seg.get("words", []) or []
    if not text:
        return [seg]

    blue, red = fighter_names
    hint = ""
    if blue or red:
        hint = (
            f"Fighter name spelling reference - Blue: {blue or '(unknown)'}, Red: {red or '(unknown)'}. "
            "If ASR sounds like these names, correct spelling.\n\n"
        )

    user_content = (
        f"{hint}"
        "Split this boxing commentary chunk into self-contained sentences.\n\n"
        "RULES:\n"
        "1) ACTION-ANALYSIS SPLIT (highest priority): When a sentence combines immediate visual action "
        "with tactical intent/analysis (often joined by comma, 'trying to', 'looking to', 'as he/she tries'), split them.\n"
        "2) VISUAL-SHIFT SPLIT: Split when moving from background/story to live action.\n"
        "3) PRESERVE INTEGRITY: Keep multiple phrases that describe ONE specific combo or action together.\n"
        "4) NO DANGLING FRAGMENTS: Every resulting sentence must be a complete thought.\n"
        "5) Preserve wording: keep lexical content and order; no paraphrase.\n"
        "6) ASR CORRECTION: Fix obvious boxing homophones ('job'->'jab', 'fainting'->'feinting').\n"
        "7) If already clean single sentence, return one-item array.\n\n"
        f"Chunk: {json.dumps(text, ensure_ascii=False)}"
    )

    try:
        resp = _safe_chat_completion(
            client,
            SPLIT_MODEL,
            [{"role": "system", "content": SPLIT_SYSTEM}, {"role": "user", "content": user_content}],
            temperature=0,
        )
        raw = resp.choices[0].message.content
        parts = json.loads(extract_json(raw or "[]"))
        if not isinstance(parts, list) or not parts:
            return [seg]
    except Exception:
        return [seg]

    if len(parts) == 1:
        ns = dict(seg)
        ns["words"] = words
        _refresh_segment(ns, fps)
        ns["text"] = str(parts[0]).strip() or ns.get("text", "")
        return [ns]

    assigned = _assign_words_to_parts(words, parts)
    out = []
    for idx_part, part_words in enumerate(assigned):
        if not part_words:
            continue
        ns = dict(seg)
        ns["words"] = part_words
        _refresh_segment(ns, fps)
        corr_text = str(parts[idx_part]).strip() if idx_part < len(parts) else ""
        if corr_text:
            ns["text"] = corr_text
        out.append(ns)
    return out if out else [seg]


def gpt_correct_and_split_segments(client: OpenAI, segments: List[Dict[str, Any]], fps: float, fighter_names: Tuple[str, str]):
    print(f"[STEP2] GPT correction+split on {len(segments)} segment(s) …")
    out = []
    for seg in tqdm(segments, desc="GPT correction+split"):
        out.extend(_split_one_segment(client, seg, fps, fighter_names))
    print(f"[STEP2] {len(segments)} → {len(out)} segments after GPT correction+split.")
    return out


def split_segments_on_pauses(segments: List[Dict[str, Any]], fps: float, pause_threshold: float, min_words: int):
    if pause_threshold <= 0:
        return segments
    out = []
    changed = 0
    for seg in segments:
        words = seg.get("words", []) or []
        if len(words) < (min_words + 1):
            out.append(seg)
            continue
        bps = []
        for j in range(1, len(words)):
            g = float(words[j].get("start_time", 0.0) or 0.0) - float(words[j - 1].get("end_time", 0.0) or 0.0)
            if g >= pause_threshold:
                bps.append(j)
        if not bps:
            out.append(seg)
            continue
        chunks, start = [], 0
        for bp in bps:
            c = words[start:bp]
            if c:
                chunks.append(c)
            start = bp
        if start < len(words):
            chunks.append(words[start:])
        merged = []
        for c in chunks:
            tok_n = sum(1 for w in c if _normalize_token(w.get("word", "")))
            if merged and tok_n < min_words:
                merged[-1].extend(c)
            else:
                merged.append(c)
        if len(merged) <= 1:
            out.append(seg)
            continue
        changed += 1
        for c in merged:
            ns = dict(seg)
            ns["words"] = c
            _refresh_segment(ns, fps)
            out.append(ns)
    if changed:
        print(f"[PAUSE_SPLIT] {changed} segment(s) split on pauses ({pause_threshold:.2f}s).")
        print(f"[PAUSE_SPLIT] {len(segments)} → {len(out)} segments.")
    return out


# ───────────────────────── Realign + fixes ─────────────────────────
def _flatten_source_words(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for s in raw_segments:
        for w in s.get("words", []) or []:
            if "start_time" in w and "end_time" in w:
                words.append(dict(w))
    words.sort(key=lambda x: float(x.get("start_time", 0.0) or 0.0))
    return words


def _realign_words_to_sentences(segs: List[Dict[str, Any]], source_words: List[Dict[str, Any]], fps: float) -> None:
    src_tokens = [_normalize_token(w.get("word", "")) for w in source_words]
    sent_counts = [len(_sent_tokens(s.get("text", ""))) for s in segs]
    p = 0
    n = len(segs)
    for i, s in enumerate(segs):
        target = _sent_tokens(s.get("text", ""))
        if i == n - 1:
            s["words"] = source_words[p:]
            _refresh_segment(s, fps)
            break
        if not target:
            s["words"] = []
            _refresh_segment(s, fps)
            continue
        remain_min = sum(1 for k in range(i + 1, n) if sent_counts[k] > 0)
        max_possible = max(0, len(source_words) - p - remain_min)
        if max_possible <= 0:
            s["words"] = []
            _refresh_segment(s, fps)
            continue
        tlen = len(target)
        min_len = max(1, int(0.5 * tlen))
        max_len = min(max_possible, int(1.8 * tlen) + 8)
        if max_len < min_len:
            min_len = max_len
        best_len, best_score = min_len, -1e9
        for L in range(min_len, max_len + 1):
            cand = src_tokens[p:p + L]
            score = _sim(target, cand) - 0.02 * abs(L - tlen)
            if score > best_score:
                best_score, best_len = score, L
        s["words"] = source_words[p:p + best_len]
        p += best_len
        _refresh_segment(s, fps)


def _timing_fix(segs: List[Dict[str, Any]], fps: float) -> None:
    # pass 1: cap long word duration
    for seg in segs:
        words = seg.get("words", []) or []
        words.sort(key=lambda w: float(w.get("start_time", 0.0) or 0.0))
        for wi, w in enumerate(words):
            ws = float(w.get("start_time", 0.0) or 0.0)
            we = float(w.get("end_time", ws) or ws)
            if (we - ws) > MAX_WORD_DURATION:
                if wi == 0:
                    w["start_time"] = round(float(int(we // 1)), 3)
                else:
                    w["end_time"] = round(ws + MAX_WORD_DURATION, 3)
        seg["words"] = words
        _refresh_segment(seg, fps)

    # pass 2: head/tail isolated word move/delete
    i = 0
    while i < len(segs):
        seg = segs[i]
        words = seg.get("words", []) or []
        if len(words) < 2:
            i += 1
            continue
        words.sort(key=lambda w: float(w.get("start_time", 0.0) or 0.0))

        head_gap = float(words[1].get("start_time", 0.0) or 0.0) - float(words[0].get("end_time", 0.0) or 0.0)
        if head_gap >= LONG_GAP_SEC:
            fw = words[0]
            moved = False
            if i > 0:
                pws = segs[i - 1].get("words", []) or []
                if pws:
                    pws.sort(key=lambda w: float(w.get("start_time", 0.0) or 0.0))
                    d = float(fw.get("end_time", 0.0) or 0.0) - float(pws[-1].get("end_time", 0.0) or 0.0)
                    if 0.0 <= d <= MOVE_TOL_SEC:
                        pws.append(fw)
                        segs[i - 1]["words"] = pws
                        _refresh_segment(segs[i - 1], fps)
                        words = words[1:]
                        moved = True
            if not moved:
                words = words[1:]

        if not words:
            segs.pop(i)
            continue

        if len(words) >= 2:
            tail_gap = float(words[-1].get("start_time", 0.0) or 0.0) - float(words[-2].get("end_time", 0.0) or 0.0)
            if tail_gap >= LONG_GAP_SEC:
                lw = words[-1]
                moved = False
                if i + 1 < len(segs):
                    nws = segs[i + 1].get("words", []) or []
                    if nws:
                        nws.sort(key=lambda w: float(w.get("start_time", 0.0) or 0.0))
                        d = abs(float(nws[0].get("start_time", 0.0) or 0.0) - float(lw.get("start_time", 0.0) or 0.0))
                        if d <= MOVE_TOL_SEC:
                            nws.insert(0, lw)
                            segs[i + 1]["words"] = nws
                            _refresh_segment(segs[i + 1], fps)
                            words = words[:-1]
                            moved = True
                if not moved:
                    words = words[:-1]

        seg["words"] = words
        if words:
            _refresh_segment(seg, fps)
            i += 1
        else:
            segs.pop(i)


def unify_boundary_timing_postprocess(segs: List[Dict[str, Any]], raw_segments: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    """Step 3: combine pause split + global realign + timing fix."""
    segs2 = split_segments_on_pauses(segs, fps, PAUSE_SPLIT_THRESHOLD, PAUSE_SPLIT_MIN_WORDS)
    source_words = _flatten_source_words(raw_segments)
    _realign_words_to_sentences(segs2, source_words, fps)
    _timing_fix(segs2, fps)
    return segs2


# ───────────────────────── Mid-gap split ─────────────────────────
def _middle_gap_candidates(words: List[Dict[str, Any]], threshold: float) -> List[Tuple[int, float]]:
    if len(words) < 3:
        return []
    cands = []
    for j in range(1, len(words) - 1):
        g = float(words[j].get("start_time", 0.0) or 0.0) - float(words[j - 1].get("end_time", 0.0) or 0.0)
        if g >= threshold:
            cands.append((j, g))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands


def _to_class(v: Any, fallback: int) -> int:
    try:
        x = int(v)
        if x in (1, 2, 3):
            return x
    except Exception:
        pass
    return fallback


def _mid_gap_split(client: OpenAI, segs: List[Dict[str, Any]], fps: float, threshold: float, max_candidates: int) -> None:
    i = 0
    while i < len(segs):
        seg = segs[i]
        words = seg.get("words", []) or []
        cands = _middle_gap_candidates(words, threshold)
        if not cands:
            i += 1
            continue

        prev_text = segs[i - 1].get("text", "") if i > 0 else ""
        next_text = segs[i + 1].get("text", "") if i + 1 < len(segs) else ""
        prev2_text = segs[i - 2].get("text", "") if i > 1 else ""
        next2_text = segs[i + 2].get("text", "") if i + 2 < len(segs) else ""

        did_split = False
        for split_idx, gap in cands[:max_candidates]:
            left_word = str(words[split_idx - 1].get("word", ""))
            right_word = str(words[split_idx].get("word", ""))
            left_local = " ".join(str(w.get("word", "")) for w in words[max(0, split_idx - 6):split_idx])
            right_local = " ".join(str(w.get("word", "")) for w in words[split_idx:min(len(words), split_idx + 6)])

            user = {
                "current_text": seg.get("text", ""),
                "current_class": seg.get("class"),
                "gap_seconds": round(gap, 3),
                "boundary_hint": f"split between '{left_word}' and '{right_word}'",
                "left_local_words": left_local,
                "right_local_words": right_local,
                "prev2_text": prev2_text,
                "prev_text": prev_text,
                "next_text": next_text,
                "next2_text": next2_text,
            }

            try:
                resp = _safe_chat_completion(
                    client,
                    CLASSIFY_MODEL,
                    [{"role": "system", "content": MID_SPLIT_SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                    temperature=0,
                )
                out = json.loads(extract_json(resp.choices[0].message.content or "{}"))
            except Exception:
                continue

            if not bool(out.get("split", False)):
                continue

            left_words = words[:split_idx]
            right_words = words[split_idx:]
            if not left_words or not right_words:
                continue

            left_seg = dict(seg)
            right_seg = dict(seg)
            left_seg["words"] = left_words
            right_seg["words"] = right_words
            _refresh_segment(left_seg, fps)
            _refresh_segment(right_seg, fps)

            old_cls = _to_class(seg.get("class"), 3)
            left_seg["class"] = _to_class(out.get("left_class"), old_cls)
            right_seg["class"] = _to_class(out.get("right_class"), old_cls)
            segs[i:i + 1] = [left_seg, right_seg]
            did_split = True
            i += 2
            break

        if not did_split:
            i += 1


# ───────────────────────── Classification ─────────────────────────
def classify_segments(client: OpenAI, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    chunks = list(range(0, len(segments), CLASSIFY_BATCH_SIZE))
    for bs in tqdm(chunks, desc="Classifying"):
        be = min(bs + CLASSIFY_BATCH_SIZE, len(segments))
        batch = segments[bs:be]

        ps = max(0, bs - CONTEXT_BEFORE)
        pe = min(len(segments), be + CONTEXT_AFTER)
        pre = segments[ps:bs]
        post = segments[be:pe]

        ctx = []
        if pre:
            ctx.append("### Context BEFORE (reference only):")
            for s in pre:
                ctx.append(f'[{s.get("start_frame","?")}–{s.get("end_frame","?")}] "{s.get("text","")}"')
            ctx.append("")
        if post:
            ctx.append("### Context AFTER (reference only):")
            for s in post:
                ctx.append(f'[{s.get("start_frame","?")}–{s.get("end_frame","?")}] "{s.get("text","")}"')
            ctx.append("")

        user_content = (
            f"{CLASS_DEF}\n---\n"
            "Classify ONLY the listed sentences. Return JSON array in same order with fields:\n"
            "text,start_frame,end_frame,start_time,end_time,class(1|2|3).\n"
            "Return JSON only.\n\n"
            + "\n".join(ctx)
            + "\n### [CLASSIFY THESE]:\n"
            + json.dumps(batch, ensure_ascii=False)
        )

        resp = _safe_chat_completion(
            client,
            CLASSIFY_MODEL,
            [{"role": "system", "content": CLASS_SYSTEM}, {"role": "user", "content": user_content}],
            temperature=0,
        )
        arr = json.loads(extract_json(resp.choices[0].message.content or "[]"))
        out.extend(arr)
    return out


# ───────────────────────── Main per-file pipeline ─────────────────────────
def process_file(client: OpenAI, path: str) -> Dict[str, Any]:
    data = json.load(open(path, "r", encoding="utf-8"))
    video_id = data.get("video_id", "")
    fps = float(data.get("fps") or 50.0)
    fighter_names = _parse_fighter_names(video_id)

    raw_segments = [
        {
            "text": (s.get("text", "") or "").strip(),
            "start_frame": s.get("start_frame"),
            "end_frame": s.get("end_frame"),
            "start_time": s.get("start_time"),
            "end_time": s.get("end_time"),
            "words": s.get("words", []) or [],
        }
        for s in data.get("segments", [])
        if (s.get("text", "") or "").strip()
    ]

    # A) Step 2: GPT correction + sentence split
    segs = gpt_correct_and_split_segments(client, raw_segments, fps, fighter_names)

    # B) Step 3: unified boundary/timing post-process (pause split + realign + timing fix)
    segs = unify_boundary_timing_postprocess(segs, raw_segments, fps)

    # C) Step 4: LLM middle-gap split
    if ENABLE_MID_GAP_SPLIT:
        _mid_gap_split(client, segs, fps, MID_GAP_THRESHOLD, MID_GAP_MAX_CANDIDATES)

    # D) Step 5: classify
    payload = [{k: v for k, v in s.items() if k != "words"} for s in segs]
    cls = classify_segments(client, payload)

    final = []
    for s, c in zip(segs, cls):
        e = dict(c)
        e["words"] = s.get("words", [])
        final.append(e)

    return {
        "video_id": data.get("video_id"),
        "file_path": data.get("file_path"),
        "fps": data.get("fps"),
        "total_frames": data.get("total_frames"),
        "language": data.get("language"),
        "classified_segments": final,
    }


# ───────────────────────── CLI ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Final self-contained commentary pipeline.")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--input_dir", default=INPUT_DIR)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    all_files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.json")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    if args.files:
        targets = [os.path.join(args.input_dir, f) for f in args.files]
    elif args.start is not None and args.end is not None:
        targets = all_files[args.start: args.end + 1]
    else:
        targets = all_files

    client = build_client()
    for p in targets:
        if not os.path.exists(p):
            print(f"[MISS] {p}")
            continue
        fn = os.path.basename(p)
        try:
            result = process_file(client, p)
            out = os.path.join(args.output_dir, fn)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[OK] {fn} -> {out}")
            write_log(f"Successfully processed {fn}")
        except Exception as e:
            msg = f"{fn}: {str(e)[:300]}"
            print(f"[FAIL] {msg}")
            write_log(msg, is_error=True)


if __name__ == "__main__":
    main()
