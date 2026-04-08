# Data Layout

Use the following directory structure after downloading the released assets:

```text
data/
├── train/
│   ├── videos/
│   ├── events/
│   └── asr/
└── eval/
    ├── videos/
    ├── events/
    └── asr/
```

Split rule:

- `train`: video id `< 478`
- `eval`: video id `>= 478`

Expected contents:

- `videos/`: match videos in `.mp4` format
- `events/`: one folder per match, each folder containing both the skeleton `.pkl` file and `video_event_inference_3.json`
- `asr/`: processed ASR JSON files with `classified_segments`

Example for `id=478`:

```text
data/eval/videos/478_LUTSAIKHAN-Altantsetseg_MINAKSHI-Minakshi.mp4
data/eval/events/478_LUTSAIKHAN-Altantsetseg_MINAKSHI-Minakshi/
├── 478_LUTSAIKHAN-Altantsetseg_MINAKSHI-Minakshi_chmr_phalp_p2d_uve.pkl
└── video_event_inference_3.json
data/eval/asr/478.json
```
