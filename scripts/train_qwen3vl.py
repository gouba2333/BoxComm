#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_vl_utils import process_vision_info
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


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
DEFAULT_TRAIN_JSONL = "/mnt/nfs_share/wangkw/streaming-vlm/sft_qwen3vl_train.jsonl"
DEFAULT_LOCAL_VIDEO_DIR = "/home/wangkw/dataset/shijs_processed"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_instruction(with_class: bool, target_class: int) -> str:
    if not with_class:
        return "Generate one natural commentary sentence for the current moment."
    class_name = CLASS_NAME_MAP.get(int(target_class), "Class 1 (Play-by-Play)")
    return (
        f"{CLASS_DEFINITION_TEXT}\n"
        f"Target class for this sample: {class_name}.\n"
        f"Generate one natural {class_name} commentary sentence for the current moment."
    )


def _find_lora_target_modules(model: nn.Module, include_vision: bool = False) -> List[str]:
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        low = name.lower()
        if "lm_head" in low:
            continue
        if not include_vision:
            if any(x in low for x in ["vision", "visual", "img", "merger", "projector"]):
                continue
        leaf = name.split(".")[-1]
        targets.add(leaf)

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    chosen = [x for x in preferred if x in targets]
    if chosen:
        return chosen
    return sorted(targets)


def _normalize_video_kwargs(video_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (video_kwargs or {}).items():
        if isinstance(v, list) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


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


@dataclass
class SampleConfig:
    k_sec: float
    post_sec: float
    input_fps: float
    nframes: int
    video_max_pixels: int
    video_min_pixels: int
    resized_height: int
    resized_width: int
    use_local_video: bool
    local_video_dir: str
    max_seq_len: int
    with_class: bool
    num_prev_events: int


class Qwen3VLSFTDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], processor: AutoProcessor, cfg: SampleConfig):
        self.rows = rows
        self.processor = processor
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def _build_messages(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        video_path = _resolve_video_path(
            str(row["video_path"]),
            use_local_video=bool(self.cfg.use_local_video),
            local_video_dir=str(self.cfg.local_video_dir),
        )
        t_mid = float(row["t_mid"])
        t0 = max(0.0, t_mid - self.cfg.k_sec)
        t1 = t_mid + self.cfg.post_sec

        target_class = int(row.get("target_class", 1))
        previous_text = str(row.get("previous_text", "")).strip() or "(none)"

        prev_events_raw = row.get("previous_events", [])
        if not isinstance(prev_events_raw, list):
            prev_events_raw = []
        prev_events = [str(x).strip() for x in prev_events_raw if str(x).strip()]
        if self.cfg.num_prev_events >= 0:
            prev_events = prev_events[-int(self.cfg.num_prev_events):]
        use_prev_events = self.cfg.num_prev_events != 0 and len(prev_events) > 0
        previous_events_text = " | ".join(prev_events) if use_prev_events else ""

        instruction_text = _build_instruction(self.cfg.with_class, target_class)

        if use_prev_events:
            if target_class == 1:
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

        video_content: Dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "start": t0,
            "end": t1,
        }
        if self.cfg.nframes > 0:
            video_content["nframes"] = int(self.cfg.nframes)
        else:
            video_content["fps"] = float(self.cfg.input_fps)

        if self.cfg.video_max_pixels is not None and int(self.cfg.video_max_pixels) > 0:
            video_content["max_pixels"] = int(self.cfg.video_max_pixels)
        if self.cfg.video_min_pixels is not None and int(self.cfg.video_min_pixels) > 0:
            video_content["min_pixels"] = int(self.cfg.video_min_pixels)
        if self.cfg.resized_height is not None and int(self.cfg.resized_height) > 0:
            video_content["resized_height"] = int(self.cfg.resized_height)
        if self.cfg.resized_width is not None and int(self.cfg.resized_width) > 0:
            video_content["resized_width"] = int(self.cfg.resized_width)

        user_text = (
            f"Previous commentary: {previous_text}\n"
            + (
                "Previous punch events (context only): "
                f"{previous_events_text}\n"
                if use_prev_events
                else ""
            )
            + f"Current time: {t_mid:.2f}s\n"
            + event_guidance
            + instruction_text
            + " Use 6-20 words."
            + " Output text only."
        )

        target_text = str(row.get("target_text", "")).strip()
        if not target_text:
            target_text = "No commentary."

        return [
            {"role": "system", "content": "You are a live boxing commentator."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    video_content,
                ],
            },
            {"role": "assistant", "content": target_text},
        ]

    def _tokenize_sample(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_messages = messages[:-1]

        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        full_text = full_text if isinstance(full_text, str) else full_text[0]
        prompt_text = prompt_text if isinstance(prompt_text, str) else prompt_text[0]

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        video_kwargs = _normalize_video_kwargs(video_kwargs or {})

        proc_kwargs: Dict[str, Any] = {
            "text": [full_text],
            "padding": False,
            "truncation": True,
            "max_length": int(self.cfg.max_seq_len),
            "return_tensors": "pt",
        }
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
            proc_kwargs.update(video_kwargs)

        full_inputs = self.processor(**proc_kwargs)

        prompt_kwargs: Dict[str, Any] = {
            "text": [prompt_text],
            "padding": False,
            "truncation": True,
            "max_length": int(self.cfg.max_seq_len),
            "return_tensors": "pt",
        }
        if image_inputs:
            prompt_kwargs["images"] = image_inputs
        if video_inputs:
            prompt_kwargs["videos"] = video_inputs
            prompt_kwargs.update(video_kwargs)

        prompt_inputs = self.processor(**prompt_kwargs)

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        prompt_len = min(prompt_len, int(labels.shape[0]))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            if key in full_inputs:
                out[key] = full_inputs[key]

        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        last_err: Optional[Exception] = None
        for _ in range(5):
            row = self.rows[idx]
            try:
                messages = self._build_messages(row)
                return self._tokenize_sample(messages)
            except Exception as e:
                last_err = e
                idx = random.randint(0, len(self.rows) - 1)
        raise RuntimeError(f"Failed to load sample after retries. Last error: {last_err}")


class DataCollatorQwen3VL:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
            vals = [f[key] for f in features if key in f]
            if len(vals) == len(features) and len(vals) > 0:
                batch[key] = torch.cat(vals, dim=0)

        return batch


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _save_args_snapshot(args, args.args_log_dir, tag="train_qwen3vl")

    dtype = torch.bfloat16 if (torch.cuda.is_available() and args.bf16) else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=args.local_files_only)
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.target_modules.strip():
        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    else:
        target_modules = _find_lora_target_modules(model, include_vision=args.lora_include_vision)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows = _read_jsonl(args.train_jsonl)
    if args.max_samples is not None and args.max_samples > 0:
        rows = rows[: int(args.max_samples)]

    cfg = SampleConfig(
        k_sec=float(args.k_sec),
        post_sec=float(args.post_sec),
        input_fps=float(args.input_fps),
        nframes=int(args.nframes),
        video_max_pixels=int(args.video_max_pixels),
        video_min_pixels=int(args.video_min_pixels),
        resized_height=int(args.resized_height),
        resized_width=int(args.resized_width),
        use_local_video=bool(args.use_local_video),
        local_video_dir=str(args.local_video_dir),
        max_seq_len=int(args.max_seq_len),
        with_class=bool(args.with_class),
        num_prev_events=int(args.num_prev_events),
    )

    train_dataset = Qwen3VLSFTDataset(rows=rows, processor=processor, cfg=cfg)
    collator = DataCollatorQwen3VL(pad_token_id=int(processor.tokenizer.pad_token_id))

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=float(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=int(args.save_total_limit),
        save_strategy=args.save_strategy,
        eval_strategy="no",
        do_eval=False,
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        remove_unused_columns=False,
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_pin_memory=bool(args.dataloader_pin_memory),
        dataloader_persistent_workers=bool(args.dataloader_persistent_workers),
        dataloader_prefetch_factor=(int(args.dataloader_prefetch_factor) if int(args.dataloader_num_workers) > 0 else None),
        report_to="none",
        optim=args.optim,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LoRA SFT for Qwen3-VL on train_streamingvlm_base.jsonl")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train_jsonl", type=str, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--output_dir", type=str, default="qwen3vl_lora_ckpt")

    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_strategy", type=str, choices=["steps", "epoch"], default="steps")
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)

    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optim", type=str, default="adamw_torch")

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_include_vision", action="store_true", default=False)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="",
        help="Comma separated module leaf names. Empty means auto-detect.",
    )

    parser.add_argument("--k_sec", type=float, default=4.0)
    parser.add_argument("--post_sec", type=float, default=0.0)
    parser.add_argument("--input_fps", type=float, default=2.0)
    parser.add_argument("--nframes", type=int, default=8)
    parser.add_argument(
        "--video_max_pixels",
        type=int,
        default=307328,
        help="Upper bound of pixels per sampled video frame (qwen-vl-utils effective cap is 602112).",
    )
    parser.add_argument(
        "--video_min_pixels",
        type=int,
        default=0,
        help="Lower bound of pixels per sampled video frame. 0 means disabled.",
    )
    parser.add_argument(
        "--resized_height",
        type=int,
        default=0,
        help="Optional fixed resize height for sampled video frames. 0 means disabled.",
    )
    parser.add_argument(
        "--resized_width",
        type=int,
        default=0,
        help="Optional fixed resize width for sampled video frames. 0 means disabled.",
    )
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--with_class", action="store_true", default=False)
    parser.add_argument("--num_prev_events", type=int, default=16)
    parser.add_argument("--use_local_video", action="store_true", default=False)
    parser.add_argument("--local_video_dir", type=str, default=DEFAULT_LOCAL_VIDEO_DIR)
    parser.add_argument("--args_log_dir", type=str, default="output_logs")

    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()
    run(args)
