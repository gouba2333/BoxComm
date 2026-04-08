#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Set, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


CLASS_NAME_MAP = {
    1: "Class 1 (Play-by-Play)",
    2: "Class 2 (Tactical)",
    3: "Class 3 (Contextual)",
}

CLASS_DEFINITION_TEXT = (
    "Class definitions:\n"
    "- Class 1 (Play-by-Play): Describe observable actions/events happening now.\n"
    "- Class 2 (Tactical): Analyze strategy, positioning, and intent.\n"
    "- Class 3 (Contextual): Provide background, history, or broader match context."
)

DEFAULT_MODEL_PATH = "/mnt/nfs_share/wangkw/.cache/huggingface/hub/Qwen3-VL-8B-Instruct"
DEFAULT_LOCAL_VIDEO_DIR = "/home/wangkw/dataset/shijs_processed"


def _resolve_video_path(video_path: str, use_local_video: bool, local_video_dir: str) -> str:
    if not use_local_video:
        return video_path
    local_path = os.path.join(local_video_dir, os.path.basename(video_path))
    if os.path.exists(local_path):
        return local_path
    return video_path


def _save_args_snapshot(args: argparse.Namespace, args_log_dir: str, tag: str) -> str:
    os.makedirs(args_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(args_log_dir, f"{tag}_pid{os.getpid()}_{ts}.args.json")
    with open(out_path, "w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["timestamp"] = ts
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _clean_text(x: str) -> str:
    y = (x or "").strip()
    y = re.sub(r"^\s*(\[?class\s*[123].*?\]?\s*:?|play-by-play\s*:|tactical\s*:|contextual\s*:)", "", y, flags=re.I)
    y = y.strip(" \n\t\"'“”")
    return y


def _read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _sample_key(row: Dict) -> Tuple:
    return (
        int(row.get("video_id", -1)),
        int(row.get("segment_index", -1)),
        round(float(row.get("t_mid", -1.0)), 2),
        str(row.get("target_text", "")).strip(),
    )


def _load_processed_keys(output_jsonl: str) -> Set[Tuple]:
    if not os.path.exists(output_jsonl):
        return set()
    keys = set()
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                keys.add(_sample_key(obj))
            except Exception:
                pass
    return keys


def run(args):
    model_path = str(args.model_path).strip()
    lora_path = str(args.lora_path).strip()

    # Common misconfiguration: passing a PEFT/LoRA checkpoint to --model_path.
    # If detected and --lora_path is empty, auto-route it to LoRA and use base model.
    if model_path and os.path.isdir(model_path):
        has_hf_config = os.path.exists(os.path.join(model_path, "config.json"))
        has_adapter_config = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if (not has_hf_config) and has_adapter_config and (not lora_path):
            if os.path.isdir(DEFAULT_MODEL_PATH) and os.path.exists(os.path.join(DEFAULT_MODEL_PATH, "config.json")):
                print(
                    "[infer_qwen3vl] Detected LoRA adapter checkpoint in --model_path; "
                    "auto-using it as --lora_path and falling back to base model:",
                    DEFAULT_MODEL_PATH,
                )
                args.lora_path = model_path
                args.model_path = DEFAULT_MODEL_PATH
            else:
                raise ValueError(
                    "Detected a LoRA adapter checkpoint in --model_path (adapter_config.json found, "
                    "config.json missing). Please pass a base model directory to --model_path and "
                    "pass this checkpoint via --lora_path."
                )

    vllm_mod = importlib.import_module("vllm")
    LLM = getattr(vllm_mod, "LLM")
    SamplingParams = getattr(vllm_mod, "SamplingParams")

    LoRARequest = None
    if args.lora_path:
        lora_mod = importlib.import_module("vllm.lora.request")
        LoRARequest = getattr(lora_mod, "LoRARequest")

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # vLLM picks devices via CUDA_VISIBLE_DEVICES.
    # Respect external setting if present; otherwise use --gpu.
    if torch.cuda.is_available() and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    llm_kwargs = {
        "model": args.model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16" if torch.cuda.is_available() else "float32",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = int(args.max_lora_rank)

    llm = LLM(**llm_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

    lora_request = None
    if args.lora_path:
        lora_name = str(args.lora_name).strip() or os.path.basename(os.path.normpath(args.lora_path)) or "qwen3vl_lora"
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=int(args.lora_id),
            lora_path=args.lora_path,
        )

    rows = _read_jsonl(args.input_jsonl)
    _save_args_snapshot(args, args.args_log_dir, tag="infer_qwen3vl")
    processed = _load_processed_keys(args.output_jsonl) if (not args.no_resume) else set()
    mode = "a" if (not args.no_resume) else "w"

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, mode, encoding="utf-8") as fout:
        for i, row in enumerate(tqdm(rows, desc="qwen3vl infer")):
            key = _sample_key(row)
            if key in processed:
                continue
            if args.debug_limit is not None and i >= args.debug_limit:
                break

            try:
                video_path = _resolve_video_path(
                    str(row["video_path"]),
                    use_local_video=bool(args.use_local_video),
                    local_video_dir=str(args.local_video_dir),
                )
                t_mid = float(row["t_mid"])
                t0 = max(0.0, t_mid - args.k_sec)
                t1 = t_mid + args.post_sec

                cls = int(row.get("target_class", 1))
                class_name = CLASS_NAME_MAP.get(cls, "Class 1 (Play-by-Play)")
                previous_text = str(row.get("previous_text", "")).strip() or "(none)"
                prev_events_raw = row.get("previous_events", [])
                if not isinstance(prev_events_raw, list):
                    prev_events_raw = []
                prev_events = [str(x).strip() for x in prev_events_raw if str(x).strip()]
                if args.num_prev_events is not None and args.num_prev_events >= 0:
                    prev_events = prev_events[-int(args.num_prev_events):]
                use_prev_events = (args.num_prev_events is None or args.num_prev_events > 0) and len(prev_events) > 0
                previous_events_text = " | ".join(prev_events) if use_prev_events else ""

                if args.with_class:
                    instruction_text = (
                        f"{CLASS_DEFINITION_TEXT}\n"
                        f"Target class for this sample: {class_name}.\n"
                        f"Generate one natural {class_name} commentary sentence for the current moment."
                    )
                else:
                    instruction_text = "Generate one natural commentary sentence for the current moment."

                if use_prev_events:
                    if cls == 1:
                        event_guidance = (
                            "Event usage rules:\n"
                            "- Previous punch events are soft context and may be noisy.\n"
                            "- Prioritize what is visible in the current clip over previous events.\n"
                            "- Use event details only when consistent with current visuals.\n"
                            "- Do NOT output repetitive generic templates; be specific to this moment.\n"
                        )
                    else:
                        event_guidance = (
                            "Event usage rules:\n"
                            "- Previous punch events are optional context only.\n"
                            "- Keep focus on current clip and target class style.\n"
                        )
                else:
                    event_guidance = ""

                # qwen-vl-utils only accepts one video sampling control at a time.
                # Passing both `fps` and `nframes` triggers:
                # "Only accept either `fps` or `nframes`"
                video_content = {
                    "type": "video",
                    "video": video_path,
                    "start": t0,
                    "end": t1,
                }
                if args.nframes is not None and args.nframes > 0:
                    video_content["nframes"] = int(args.nframes)
                else:
                    video_content["fps"] = float(args.input_fps)

                messages = [
                    {"role": "system", "content": "You are a live boxing commentator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    (
                                        f"Previous commentary: {previous_text}\n"
                                        + (
                                            "Previous punch events (context only): "
                                            f"{previous_events_text}\n"
                                            if use_prev_events
                                            else ""
                                        )
                                        + f"Current time: {t_mid:.2f}s\n"
                                        + event_guidance
                                        + f"{instruction_text}"
                                        + " Use 6-20 words."
                                        + " Output text only."
                                    )
                                ),
                            },
                            video_content,
                        ],
                    },
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt = text if isinstance(text, str) else text[0]

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                )
                video_kwargs = video_kwargs or {}
                llm_input = {"prompt": prompt}

                if image_inputs:
                    llm_input["multi_modal_data"] = {"image": image_inputs}
                elif video_inputs:
                    # vLLM expects video in THWC numpy format.
                    vid = video_inputs[0]
                    if isinstance(vid, torch.Tensor):
                        vid_nd = vid.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    else:
                        vid_nd = vid
                    _raw_fps = video_kwargs.get("fps", None)
                    actual_fps = float(_raw_fps[0]) if isinstance(_raw_fps, list) else (
                        float(_raw_fps) if _raw_fps is not None else float(args.input_fps)
                    )
                    _nf = int(vid_nd.shape[0]) if hasattr(vid_nd, "shape") else 0
                    video_metadata = {
                        "fps": actual_fps,
                        "total_num_frames": _nf,
                        "frames_indices": list(range(_nf)),
                    }
                    llm_input["multi_modal_data"] = {"video": (vid_nd, video_metadata)}

                sampling_params = SamplingParams(
                    temperature=0.0 if args.temperature <= 0 else args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                    repetition_penalty=1.05,
                )

                if lora_request is None:
                    outputs = llm.generate([llm_input], sampling_params=sampling_params)
                else:
                    outputs = llm.generate([llm_input], sampling_params=sampling_params, lora_request=lora_request)
                pred_raw = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
                pred_text = _clean_text(pred_raw)

                rec = {
                    "video_id": row.get("video_id"),
                    "segment_index": row.get("segment_index"),
                    "t_mid": row.get("t_mid"),
                    "target_class": cls,
                    "target_text": row.get("target_text"),
                    "pred_text": pred_text,
                    "pred_text_raw": pred_raw,
                    "model": "Qwen3-VL-8B-Instruct" + ("+LoRA" if args.lora_path else ""),
                }
            except Exception as e:
                rec = {
                    "video_id": row.get("video_id"),
                    "segment_index": row.get("segment_index"),
                    "t_mid": row.get("t_mid"),
                    "target_class": row.get("target_class"),
                    "target_text": row.get("target_text"),
                    "pred_text": "",
                    "error": str(e),
                    "model": "Qwen3-VL-8B-Instruct" + ("+LoRA" if args.lora_path else ""),
                }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--input_jsonl", type=str, default="sft_qwen3vl_eval.jsonl")
    ap.add_argument("--output_jsonl", type=str, default="infer_results_qwen3vl_vllm_full_with_class_prev16_e1_prompttuned.jsonl")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k_sec", type=float, default=4.0)
    ap.add_argument("--post_sec", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--input_fps", type=float, default=2.0)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--debug_limit", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true", default=False)
    ap.add_argument("--with_class", action="store_true", default=True)
    ap.add_argument("--num_prev_events", type=int, default=16)
    ap.add_argument("--use_local_video", action="store_true", default=False)
    ap.add_argument("--local_video_dir", type=str, default=DEFAULT_LOCAL_VIDEO_DIR)
    ap.add_argument("--args_log_dir", type=str, default="output_logs")
    ap.add_argument("--lora_path", type=str, default="")
    ap.add_argument("--lora_name", type=str, default="qwen3vl_lora")
    ap.add_argument("--lora_id", type=int, default=1)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    args = ap.parse_args()
    run(args)
