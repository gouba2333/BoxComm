#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm


CLASS_NAME_MAP = {
    1: "Class 1 (Play-by-Play)",
    2: "Class 2 (Tactical)",
    3: "Class 3 (Contextual)",
}

MODEL_NAME = os.getenv("MODEL_CLASSIFY", os.getenv("MODEL_NAME", "gpt-5.2"))


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


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


def _safe_chat_completion(client: OpenAI, model: str, messages: List[Dict[str, str]], max_attempts: int = 5):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, temperature=0)
        except Exception as e:
            err = str(e)
            if "temperature" in err and any(k in err for k in ("Unsupported value", "does not support", "unsupported_value")):
                try:
                    return client.chat.completions.create(model=model, messages=messages)
                except Exception as e2:
                    last_err = e2
            else:
                last_err = e
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 10))
    raise last_err


def _key_of(row: Dict[str, Any]) -> Tuple[int, int]:
    return _safe_int(row.get("video_id", -1), -1), _safe_int(row.get("segment_index", -1), -1)


def _align_with_gt(
    pred_rows: List[Dict[str, Any]],
    gt_index: Dict[Tuple[int, int], Dict[str, Any]],
    skip_empty_pred: bool = True,
) -> Tuple[List[Dict[str, Any]], int, int]:
    aligned: List[Dict[str, Any]] = []
    missing = 0
    skipped_empty = 0
    for p in pred_rows:
        gt = gt_index.get(_key_of(p))
        if gt is None:
            missing += 1
            continue
        pred_text = str(p.get("pred_text", "") or "").strip()
        if skip_empty_pred and (not pred_text):
            skipped_empty += 1
            continue
        aligned.append(
            {
                "video_id": _safe_int(p.get("video_id", -1), -1),
                "segment_index": _safe_int(p.get("segment_index", -1), -1),
                "target_class": _safe_int(gt.get("target_class", -1), -1),
                "target_text": str(gt.get("target_text", "") or "").strip(),
                "pred_text": pred_text,
            }
        )
    return aligned, missing, skipped_empty


def _load_bertscore_metric():
    try:
        import evaluate

        return evaluate.load("bertscore")
    except Exception:
        return None


def _compute_bertscore_f1(
    bertscore_metric,
    preds: List[str],
    refs: List[str],
    lang: str = "en",
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    if bertscore_metric is None or len(preds) == 0:
        return 0.0
    # Robustness: bert-score can fail on empty strings with some tokenizer versions.
    preds = [str(x).strip() if str(x).strip() else "[empty]" for x in preds]
    refs = [str(x).strip() if str(x).strip() else "[empty]" for x in refs]
    try:
        res = bertscore_metric.compute(
            predictions=preds,
            references=refs,
            lang=lang,
            batch_size=batch_size,
            device=device,
        )
        f1 = res.get("f1", [])
        return float(np.mean(f1)) if f1 else 0.0
    except Exception as e:
        print(f"[WARN] BERTScore failed: {e}")
        return 0.0


def _build_gpt_eval_prompt(batch_rows: List[Dict[str, Any]], mode: str = "lenient") -> str:
    pairs = []
    for i, r in enumerate(batch_rows):
        pairs.append({"i": i, "reference": r["target_text"], "prediction": r["pred_text"]})

    if mode == "strict": # default is lenient, strict mode can be used for play-by-play class where we require more precise matching of core meaning and event.
        rubric = (
            "You are an evaluator for sports commentary sentence matching.\n"
            "For each pair, output score=1 only if prediction and reference describe basically the same content/event/meaning.\n"
            "Allow paraphrase and minor wording differences.\n"
            "Output score=0 if prediction is inconsistent, talks about another content, adds major unrelated claims, or misses core meaning.\n"
            "Return JSON ONLY as an array of objects: [{\"i\":0,\"score\":0|1}, ...].\n"
            "Do not add explanation."
        )
    else:
        rubric = (
            "You are an evaluator for sports commentary semantic overlap.\n"
            "For each pair, output score=1 if prediction and reference are roughly about the same action/topic.\n"
            "Be lenient to paraphrase, generic wording, missing minor details, and extra non-conflicting details.\n"
            "Output score=0 only when prediction clearly talks about a different action/topic or contradicts core meaning.\n"
            "Return JSON ONLY as an array of objects: [{\"i\":0,\"score\":0|1}, ...].\n"
            "Do not add explanation."
        )
    return rubric + "\n\nPairs:\n" + json.dumps(pairs, ensure_ascii=False)


def _parse_gpt_scores(raw: str, expected_n: int) -> List[int]:
    out = [0] * expected_n
    try:
        parsed = json.loads(_extract_json(raw))
    except Exception:
        return out

    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            if isinstance(item, dict):
                idx = _safe_int(item.get("i", i), i)
                score = _safe_int(item.get("score", 0), 0)
            else:
                idx = i
                score = _safe_int(item, 0)
            if 0 <= idx < expected_n:
                out[idx] = 1 if score == 1 else 0
    return out


def _compute_gpt_consistency(
    client: OpenAI,
    model: str,
    rows: List[Dict[str, Any]],
    batch_size: int = 120,
    mode: str = "lenient",
) -> List[int]:
    if not rows:
        return []

    scores: List[int] = []
    for st in tqdm(range(0, len(rows), batch_size), desc="GPT consistency", leave=False):
        batch = rows[st: st + batch_size]
        user_content = _build_gpt_eval_prompt(batch, mode=mode)
        resp = _safe_chat_completion(
            client,
            model,
            [
                {"role": "system", "content": "You are a strict JSON-only evaluator."},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content
        batch_scores = _parse_gpt_scores(raw, len(batch))
        scores.extend(batch_scores)
    return scores


def _subset_by_class(rows: List[Dict[str, Any]], scores: List[int], cls_id: int) -> Tuple[List[str], List[str], List[int]]:
    preds: List[str] = []
    refs: List[str] = []
    gpt_scores: List[int] = []
    for i, r in enumerate(rows):
        if _safe_int(r.get("target_class", -1), -1) != cls_id:
            continue
        refs.append(r.get("target_text", ""))
        preds.append(r.get("pred_text", ""))
        gpt_scores.append(scores[i])
    return preds, refs, gpt_scores


def evaluate_file(
    pred_file: str,
    gt_index: Dict[Tuple[int, int], Dict[str, Any]],
    bertscore_metric,
    client: OpenAI,
    bert_lang: str,
    bert_batch_size: int,
    bert_device: str,
    skip_empty_pred: bool,
    gpt_model: str,
    gpt_batch_size: int,
    gpt_mode: str,
) -> Dict[str, Any]:
    pred_rows = _read_jsonl(pred_file)
    aligned, num_missing, num_empty_skipped = _align_with_gt(
        pred_rows,
        gt_index,
        skip_empty_pred=skip_empty_pred,
    )

    preds_all = [r["pred_text"] for r in aligned]
    refs_all = [r["target_text"] for r in aligned]

    gpt_scores_all = _compute_gpt_consistency(
        client=client,
        model=gpt_model,
        rows=aligned,
        batch_size=gpt_batch_size,
        mode=gpt_mode,
    )
    if len(gpt_scores_all) != len(aligned):
        gpt_scores_all = (gpt_scores_all + [0] * len(aligned))[: len(aligned)]

    overall_bert = _compute_bertscore_f1(
        bertscore_metric,
        preds_all,
        refs_all,
        lang=bert_lang,
        batch_size=bert_batch_size,
        device=bert_device,
    )
    overall_gpt = float(np.mean(gpt_scores_all)) if gpt_scores_all else 0.0

    result: Dict[str, Any] = {
        "num_preds": len(pred_rows),
        "num_aligned": len(aligned),
        "num_missing_in_gt": num_missing,
        "num_empty_pred_skipped": num_empty_skipped,
        "overall": {
            "BERTScore_F1": overall_bert,
            "GPT_Consistency": overall_gpt,
            "GPT_Consistency_hits": int(sum(gpt_scores_all)),
        },
        "per_class": {},
    }

    for cls_id in (1, 2, 3):
        preds_c, refs_c, gpt_c = _subset_by_class(aligned, gpt_scores_all, cls_id)
        bert_c = _compute_bertscore_f1(
            bertscore_metric,
            preds_c,
            refs_c,
            lang=bert_lang,
            batch_size=bert_batch_size,
            device=bert_device,
        )
        gpt_avg_c = float(np.mean(gpt_c)) if gpt_c else 0.0
        result["per_class"][str(cls_id)] = {
            "class_name": CLASS_NAME_MAP[cls_id],
            "count": len(preds_c),
            "BERTScore_F1": bert_c,
            "GPT_Consistency": gpt_avg_c,
            "GPT_Consistency_hits": int(sum(gpt_c)),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction files with two metrics: BERTScore + GPT consistency")
    parser.add_argument("--gt_file", type=str, default="sft_qwen3vl_eval.jsonl")
    parser.add_argument(
        "--pred_files",
        nargs="+",
        default=[
            "output_logs/infer_qwen3vl_local_prev0_withclass_gpu6.jsonl",
            "output_logs/infer_qwen3vl_local_prev16_withclass_gpu6.jsonl",
        ],
    )
    parser.add_argument("--output_json", type=str, default="output_logs/eval_metrics/two_metrics_bertscore_gpt52.json")
    parser.add_argument("--bert_lang", type=str, default="en")
    parser.add_argument("--bert_batch_size", type=int, default=256)
    parser.add_argument("--bert_device", type=str, default="cpu")
    parser.add_argument("--skip_empty_pred", type=int, default=1, help="1: skip empty pred_text rows, 0: keep them")
    parser.add_argument("--gpt_model", type=str, default=MODEL_NAME)
    parser.add_argument("--gpt_batch_size", type=int, default=120)
    parser.add_argument("--gpt_mode", choices=["strict", "lenient"], default="lenient")
    args = parser.parse_args()

    gt_rows = _read_jsonl(args.gt_file)
    gt_index = {_key_of(r): r for r in gt_rows}

    bertscore_metric = _load_bertscore_metric()
    client = build_client()

    all_results: Dict[str, Any] = {}
    for pred_file in args.pred_files:
        print(f"[EVAL] {pred_file}")
        all_results[pred_file] = evaluate_file(
            pred_file=pred_file,
            gt_index=gt_index,
            bertscore_metric=bertscore_metric,
            client=client,
            bert_lang=args.bert_lang,
            bert_batch_size=args.bert_batch_size,
            bert_device=args.bert_device,
            skip_empty_pred=bool(int(args.skip_empty_pred)),
            gpt_model=args.gpt_model,
            gpt_batch_size=args.gpt_batch_size,
            gpt_mode=args.gpt_mode,
        )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
