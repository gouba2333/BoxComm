# Hugging Face Upload Playbook

This note describes a practical release order for `BoxComm-Dataset` and `BoxComm-Benchmark`.

## Recommended release order

1. publish the GitHub code repository first
2. publish `BoxComm-Benchmark`
3. publish `BoxComm-Dataset`
4. update the GitHub README with the final Hugging Face links

This order works well because the benchmark package is smaller and easier to polish first.

## What to publish to `BoxComm-Benchmark`

Recommended contents:

```text
BoxComm-Benchmark/
├── README.md
├── manifests/
│   └── benchmark_manifest_eval_v1.jsonl
├── metadata/
│   └── eval_metadata_v1.csv
├── examples/
│   ├── generation_prediction_example.jsonl
│   └── streaming_prediction_example.jsonl
└── metrics/
    └── version.txt
```

You already have these local source files:

- `benchmark/manifests/benchmark_manifest_eval_v1.jsonl`
- `benchmark/metadata/eval_metadata_v1.csv`
- `docs/huggingface_benchmark_card.md`

## What to publish to `BoxComm-Dataset`

Recommended contents:

```text
BoxComm-Dataset/
├── README.md
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

If raw videos cannot be publicly redistributed, use this alternative:

```text
BoxComm-Dataset/
├── README.md
├── train/
│   ├── events/
│   └── asr/
├── eval/
│   ├── events/
│   └── asr/
├── metadata/
└── manifests/
```

and state clearly in the dataset card that raw videos are not redistributed.

## Practical packaging suggestions

### Small and medium files

Upload directly:

- `.json`
- `.jsonl`
- `.csv`
- `.md`
- `.png`

### Event folders

If you have many small event directories, package per-video folders or per-split tar archives:

- `train_events_part01.tar`
- `eval_events.tar`

### Large videos

For large `.mp4` files:

- keep a clear split structure
- use chunked archives if the folder is too large to upload comfortably
- prefer stable, human-readable filenames

Recommended examples:

- `train_videos_part01.tar`
- `train_videos_part02.tar`
- `eval_videos.tar`

## Hugging Face upload methods

According to the official Hugging Face docs, the two most practical methods are:

1. Hub UI upload
2. Python upload with `huggingface_hub`

Official references:

- https://huggingface.co/docs/hub/en/datasets-adding
- https://huggingface.co/docs/huggingface_hub/guides/upload

## Suggested local staging folders before upload

Create two clean local folders before publishing:

```text
/path/to/hf_release/
├── BoxComm-Benchmark/
└── BoxComm-Dataset/
```

Fill them first, then upload.

This makes it much easier to check file sizes, naming, and missing metadata before publishing.
