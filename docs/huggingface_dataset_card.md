---
pretty_name: BoxComm-Dataset
license: apache-2.0
task_categories:
- text-generation
- video-to-text
tags:
- multimodal
- video
- boxing
- sports
- commentary-generation
---

# BoxComm-Dataset

BoxComm-Dataset is the official data release for BoxComm, a benchmark for category-aware boxing commentary generation and narration-rhythm evaluation.

## Resources

- Project Page: https://gouba2333.github.io/BoxComm
- Paper: http://arxiv.org/abs/2604.04419
- Code: https://github.com/gouba2333/BoxComm
- Dataset: https://huggingface.co/datasets/gouba2333/BoxComm-Dataset
- Benchmark: https://huggingface.co/datasets/gouba2333/BoxComm

## Overview

This dataset release is intended for training, analysis, and reproducible preprocessing. It contains the complete processed videos together with the released annotations and benchmark metadata.

Recommended structure:

```text
BoxComm-Dataset/
├── train/
│   ├── videos/
│   ├── events/
│   └── asr/
├── eval/
│   ├── videos/
│   ├── events/
│   └── asr/
└── metadata/
```

The split convention is:

- `train`: video id `< 478`
- `eval`: video id `>= 478`

Each event directory should contain:

- one skeleton `.pkl` file
- one `video_event_inference_3.json` file

Each ASR JSON file should contain `classified_segments`.

## What is included

- processed match videos
- event annotations
- skeleton data
- ASR with sentence segmentation
- 3-way commentary labels
- split metadata

## Intended uses

- supervised fine-tuning for commentary generation
- category-aware commentary evaluation
- narration-rhythm analysis
- multimodal sports video understanding research

## Data preparation in the code repository

The official code repository provides:

- `scripts/prep_qwen3vl_sft_data.py`
- `scripts/train_qwen3vl.py`
- `scripts/infer_qwen3vl.py`
- `scripts/eval_metrics.py`
- `scripts/eval_streaming_cls_metrics.py`

Repository: https://github.com/gouba2333/BoxComm

## Licensing

The public release includes processed videos, ASR annotations, event JSON files, skeleton PKL files, and benchmark metadata for research use.

## Citation

```bibtex
@article{wang2026boxcomm,
  title={BoxComm: Benchmarking Category-Aware Commentary Generation and Narration Rhythm in Boxing},
  author={Wang, Kaiwen and Zheng, Kaili and Deng, Rongrong and Shi, Yiming and Guo, Chenyi and Wu, Ji},
  journal={arXiv preprint arXiv:2604.04419},
  year={2026}
}
```
