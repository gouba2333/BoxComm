# BoxComm Release Checklist

## GitHub

- Add repository description
- Add website URL
- Add topics: `benchmark`, `boxing`, `video-understanding`, `multimodal`, `commentary-generation`
- Publish a `v1.0-code` release
- Keep `LICENSE` visible in the repository root

## README

- Fill in Dataset link
- Fill in Benchmark link
- Keep project page and paper links at the top
- Keep evaluation input and output formats explicit

## Hugging Face

- Create `BoxComm-Dataset`
- Create `BoxComm-Benchmark`
- Paste the prepared cards from:
  - `docs/huggingface_dataset_card.md`
  - `docs/huggingface_benchmark_card.md`
- Add file structure and schema examples
- Add version numbers and checksums if possible

## Legal and licensing

- Confirm whether raw videos can be redistributed
- If not, release annotations and derived artifacts separately
- State data license terms explicitly in the dataset and benchmark cards

## Community-facing polish

- Publish one or more GitHub Releases
- Add example prediction files
- Add one benchmark quickstart command block
