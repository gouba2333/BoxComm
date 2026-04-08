import os
import sys

# 使用 nfs_share 上已有的完整模型缓存，避免重新下载
os.environ.setdefault("HF_HOME", "/mnt/nfs_share/wangkw/.cache/huggingface")

# 将 pip 安装的 cuDNN 9 加入 LD_LIBRARY_PATH。
_CUDNN_LIB = (
    "/mnt/nfs_share/wangkw/anaconda3/envs/internvideo"
    "/lib/python3.10/site-packages/nvidia/cudnn/lib"
)
_CT2_LIB = (
    "/mnt/nfs_share/wangkw/anaconda3/envs/internvideo"
    "/lib/python3.10/site-packages/ctranslate2.libs"
)
if _CUDNN_LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_ld = ":".join(
        p for p in [_CUDNN_LIB, _CT2_LIB, os.environ.get("LD_LIBRARY_PATH", "")] if p
    )
    os.environ["LD_LIBRARY_PATH"] = new_ld
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import cv2
import torch
import glob
import argparse
import whisperx
from tqdm import tqdm

# ================= 配置区域 =================
VIDEO_DIR = "/hdd/wangkw/dataset/shijs_processed"      # 视频文件夹路径
MODEL_SIZE = "large-v3"     # 推荐 large-v3，显存不够改用 medium
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"    # cuda 用 float16，cpu 改用 int8
BATCH_SIZE = 16             # 转录 batch size，显存不足时调小
OUTPUT_ROOT = "./output_whisperx"   # 输出根目录

# 对齐模型：whisperx 需要对每种语言加载对齐模型
# 如果视频语言固定为英文，可设置 LANGUAGE = "en"，跳过自动检测加速
LANGUAGE = None             # None 表示自动检测，"en" 表示固定英文
TASK = "translate"          # "transcribe" 保留原语言，"translate" 强制输出英文

VAD_METHOD = "silero"       # "silero" 或 "pyannote"；pyannote 需要匹配的 cuDNN 版本

# HuggingFace Token（仅启用说话人分离时需要，否则留空）
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENABLE_DIARIZATION = False  # 是否开启说话人分离（需要 pyannote.audio）
# ===========================================


def get_video_info(video_path):
    """获取视频的 FPS 和总帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def build_segment_list(segments, fps, total_frames):
    """
    将 whisperx 的 segment 列表转换为统一格式，并附加帧索引信息。
    每个 segment 内若存在 word-level 时间戳，则一并保留在 'words' 字段中。
    """
    segment_list = []
    for seg in segments:
        start_time = seg.get("start", 0.0)
        end_time = seg.get("end", 0.0)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # 帧数边界检查
        if end_frame > total_frames:
            end_frame = total_frames

        segment_data = {
            "text": seg.get("text", "").strip(),
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

        # 保留词级别时间戳（whisperx align 之后才有）
        if "words" in seg:
            words = []
            for w in seg["words"]:
                word_entry = {
                    "word": w.get("word", "").strip(),
                    "start_time": round(w.get("start", 0.0), 3),
                    "end_time": round(w.get("end", 0.0), 3),
                    "score": round(w.get("score", 0.0), 3),
                }
                words.append(word_entry)
            segment_data["words"] = words

        # 保留说话人标签（diarization 之后才有）
        if "speaker" in seg:
            segment_data["speaker"] = seg["speaker"]

        segment_list.append(segment_data)
    return segment_list


def process_videos(start_idx, end_idx):
    """
    处理指定范围的视频（每个视频生成独立 JSON 文件）。
    :param start_idx: 处理起始索引（包含，按视频名称排序）
    :param end_idx:   处理结束索引（包含，按视频名称排序）
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # -------- 加载 Whisper 转录模型 --------
    print(f"正在加载转录模型: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...")
    asr_model = whisperx.load_model(
        MODEL_SIZE,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,        # None = 自动检测
        vad_method=VAD_METHOD,
    )

    # -------- 加载对齐模型缓存（懒加载，按语言缓存） --------
    align_model_cache = {}  # {language_code: (model_a, metadata)}

    # -------- 可选：说话人分离模型 --------
    diarize_model = None
    if ENABLE_DIARIZATION:
        if not HF_TOKEN:
            raise ValueError("开启说话人分离需要提供 HF_TOKEN（HuggingFace 访问令牌）")
        print("正在加载说话人分离模型...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=DEVICE,
        )

    # -------- 收集视频列表 --------
    extensions = ["*.mp4", "*.mkv", "*.avi", "*.flv"]
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_files.sort()
    total_videos = len(video_files)
    print(f"共发现 {total_videos} 个视频文件（已按名称排序）。")

    # -------- 边界检查 --------
    if start_idx < 0:
        start_idx = 0
        print("起始索引修正为 0（输入小于 0）")
    if end_idx >= total_videos:
        end_idx = total_videos - 1
        print(f"结束索引修正为 {end_idx}（超出视频总数）")
    if start_idx > end_idx:
        raise ValueError(f"起始索引 {start_idx} 大于结束索引 {end_idx}")

    target_videos = video_files[start_idx : end_idx + 1]
    print(f"本次处理范围: [{start_idx}, {end_idx}]，共 {len(target_videos)} 个视频")

    if not target_videos:
        print("没有需要处理的视频！")
        return

    # -------- 逐个处理视频 --------
    for video_path in tqdm(target_videos, desc="处理进度"):
        try:
            video_filename = os.path.basename(video_path)
            video_id = os.path.splitext(video_filename)[0]

            # 输出文件名：按 _ 分割取第一部分
            name_part = video_filename.split("_")[0] if "_" in video_filename else video_id
            output_filename = f"{name_part}.json"
            output_file_path = os.path.join(OUTPUT_ROOT, output_filename)

            # 已存在合法 JSON 则直接跳过
            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    if existing_data.get("segments") is not None:
                        print(f"\n[跳过] {video_filename} 已处理完毕")
                        continue
                except Exception as e:
                    print(f"\n[警告] 读取 {output_file_path} 失败: {e}，将重新处理")

            # 1. 获取视频元数据
            fps, total_frames = get_video_info(video_path)
            if not fps:
                print(f"\n[跳过] 无法读取视频: {video_filename}")
                continue

            # 2. 加载音频（whisperx 统一使用 numpy 数组，采样率 16kHz）
            audio = whisperx.load_audio(video_path)

            # 3. 转录（task="translate" 输出英文）
            result = asr_model.transcribe(
                audio,
                batch_size=BATCH_SIZE,
                task=TASK,
            )
            detected_language = result.get("language", "unknown")
            print(f"\n[转录完成] {video_filename}  检测语言: {detected_language}")

            # 4. 词级别强制对齐（仅当 task="transcribe" 且语言有对齐模型时有意义；
            #    task="translate" 时输出为英文，对齐仍可尝试但效果因版本而异）
            try:
                lang_key = detected_language if TASK == "transcribe" else "en"
                if lang_key not in align_model_cache:
                    print(f"  加载对齐模型: {lang_key}")
                    model_a, metadata = whisperx.load_align_model(
                        language_code=lang_key,
                        device=DEVICE,
                    )
                    align_model_cache[lang_key] = (model_a, metadata)
                model_a, metadata = align_model_cache[lang_key]
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False,
                )
                print(f"  词级别对齐完成")
            except Exception as e:
                print(f"  [警告] 词级别对齐失败（将使用段级别时间戳）: {e}")

            # 5. 可选：说话人分离
            if diarize_model is not None:
                try:
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    print(f"  说话人分离完成")
                except Exception as e:
                    print(f"  [警告] 说话人分离失败: {e}")

            # 6. 构建 segment 列表（含帧索引 & 词级别时间戳）
            segments = result.get("segments", [])
            segment_list = build_segment_list(segments, fps, total_frames)

            # 7. 构建完整数据结构
            video_data = {
                "video_id": video_id,
                "file_path": os.path.abspath(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "language": detected_language,
                "task": TASK,
                "model": MODEL_SIZE,
                "segments": segment_list,
            }

            # 8. 写入 JSON 文件
            with open(output_file_path, "w", encoding="utf-8") as f_out:
                json.dump(video_data, f_out, ensure_ascii=False, indent=2)

            print(f"  [完成] → {output_file_path}")

        except Exception as e:
            print(f"\n[错误] 处理 {video_filename} 时出错: {e}")

    print(f"\n全部完成！结果已保存至 {OUTPUT_ROOT} 目录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 WhisperX 处理指定范围的视频，每个视频生成独立 JSON 文件（含词级别时间戳）"
    )
    parser.add_argument("start", type=int, help="处理起始索引（包含，按视频名称排序）")
    parser.add_argument("end", type=int, help="处理结束索引（包含，按视频名称排序）")
    args = parser.parse_args()

    process_videos(args.start, args.end)
