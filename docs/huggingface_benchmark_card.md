# BoxComm-Benchmark

BoxComm-Benchmark is the official benchmark release for standardized BoxComm evaluation.

It is designed for fair comparison and leaderboard-style reporting, and is separate from the broader training-oriented dataset release.

## Resources

- Project Page: https://gouba2333.github.io/BoxComm
- Paper: http://arxiv.org/abs/2604.04419
- Code: https://github.com/gouba2333/BoxComm
- Dataset: add the BoxComm-Dataset link here after publishing

## Benchmark scope

This benchmark focuses on two tasks:

1. category-conditioned commentary generation
2. commentary rhythm and class-distribution evaluation

## Recommended contents

```text
BoxComm-Benchmark/
├── eval/
│   ├── manifests/
│   ├── asr/
│   ├── events/
│   └── metadata/
├── metrics/
│   └── version.txt
└── examples/
```

Recommended files:

- official eval split manifest
- official `qwen3vl_sft_eval.jsonl`
- per-video metadata for evaluation
- example prediction files
- metric version information

## Official evaluation

### Task 1: category-conditioned generation

Use:

- `scripts/eval_metrics.py`

Expected prediction format:

```json
{"video_id": 478, "segment_index": 0, "t_mid": 1.0, "pred_text": "..."}
```

### Task 2: commentary rhythm evaluation

Use:

- `scripts/eval_streaming_cls_metrics.py`

Expected prediction format:

```json
{"video_id": 478, "responses": [{"start_time": 0.0, "end_time": 1.0, "response": "..."}, ...]}
```

## Versioning

Use explicit benchmark versions such as:

- `v1.0`
- `v1.1`

and keep the official metric scripts versioned together with the benchmark release.

## Submission guidance

Recommended benchmark policy:

- use the provided eval split only
- do not tune on official eval annotations
- report the exact model, checkpoint, and prompt configuration
- report both generation metrics and rhythm metrics when applicable

## Citation

```bibtex
@article{wang2026boxcomm,
  title={BoxComm: Benchmarking Category-Aware Commentary Generation and Narration Rhythm in Boxing},
  author={Wang, Kaiwen and Zheng, Kaili and Deng, Rongrong and Shi, Yiming and Guo, Chenyi and Wu, Ji},
  journal={arXiv preprint arXiv:2604.04419},
  year={2026}
}
```
