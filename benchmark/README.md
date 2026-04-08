# BoxComm Benchmark

This directory contains the local benchmark-side assets prepared for the BoxComm public release.

## Included files

- `manifests/benchmark_manifest_eval_v1.jsonl`
  - official evaluation manifest for the current benchmark split
- `metadata/eval_metadata_v1.csv`
  - compact per-video metadata table for the official evaluation split

## Current benchmark scope

The current official evaluation split contains 40 videos with `video_id >= 478`.

The benchmark supports two evaluation settings:

1. category-conditioned commentary generation
2. commentary rhythm and class-distribution evaluation

## Related scripts

- `scripts/build_benchmark_manifest.py`
- `scripts/eval_metrics.py`
- `scripts/eval_streaming_cls_metrics.py`

## Planned public release

These files are intended to be mirrored into the Hugging Face benchmark repository:

- `BoxComm-Benchmark`

Before publishing, also copy:

- the benchmark card draft from `docs/huggingface_benchmark_card.md`
- any official example predictions
- metric version information if you want explicit versioned releases
