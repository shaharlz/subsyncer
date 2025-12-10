#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import difflib
import numpy as np
import pysubs2
import whisper


# ---------- Data types ----------

@dataclass
class Segment:
    text: str
    start: float  # seconds
    end: float    # seconds


# ---------- Utils ----------

def run_cmd(cmd: list):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        print("Command failed:", " ".join(cmd), file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise


def extract_audio(
    video_path: str,
    out_wav_path: str,
    start: float = 0.0,
    duration: float = 300.0,
    sample_rate: int = 16000,
):
    """
    Use ffmpeg to extract a mono WAV sample from the video.
    """
    cmd = [
        "ffmpeg",
        "-y",                # overwrite
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn",               # no video
        "-ac", "1",          # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        out_wav_path,
    ]
    print("[+] Extracting audio sample with ffmpeg...")
    run_cmd(cmd)
    print(f"[+] Audio sample written to {out_wav_path}")


def run_asr(audio_path: str, model_name: str = "small") -> List[Segment]:
    """
    Run Whisper ASR on the audio and return a list of segments.
    """
    print(f"[+] Loading Whisper model: {model_name} (first time may take a while)...")
    model = whisper.load_model(model_name)

    print("[+] Running ASR on audio sample...")
    result = model.transcribe(audio_path, verbose=False)

    segments = [
        Segment(text=seg["text"], start=float(seg["start"]), end=float(seg["end"]))
        for seg in result["segments"]
    ]

    print(f"[+] ASR produced {len(segments)} segments")
    return segments


def normalize_text(text: str) -> str:
    """
    Lowercase, strip HTML tags, remove most punctuation, collapse spaces.
    """
    text = text.lower()
    # remove html-like tags
    text = re.sub(r"<[^>]+>", " ", text)
    # keep letters/digits/space
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_subtitle_cues(subs: pysubs2.SSAFile, max_cues: int = 200) -> List[pysubs2.SSAEvent]:
    """
    Take up to max_cues cues from the beginning of the subtitle file.
    """
    cues = list(subs)
    return cues[:max_cues]


def match_segments_to_cues(
    segments: List[Segment],
    cues: List[pysubs2.SSAEvent],
    similarity_threshold: float = 0.6,
) -> List[Tuple[float, float, float]]:
    """
    For each subtitle cue, find the most similar ASR segment by text.
    Return list of (t_sub, t_audio, similarity).

    t_sub: subtitle time in seconds (cue start)
    t_audio: ASR time in seconds (segment center)
    """
    pairs = []

    print("[+] Matching subtitle lines to ASR segments...")
    for cue in cues:
        # pysubs2 stores milliseconds
        t_sub = cue.start / 1000.0  # seconds
        cue_text = cue.plaintext
        norm_cue = normalize_text(cue_text)

        # skip very short or empty texts (like "Hi", "...", "OK")
        if len(norm_cue) < 5:
            continue

        best_score = 0.0
        best_seg = None

        for seg in segments:
            norm_seg = normalize_text(seg.text)
            if not norm_seg:
                continue

            score = difflib.SequenceMatcher(None, norm_cue, norm_seg).ratio()
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg is not None and best_score >= similarity_threshold:
            t_audio = (best_seg.start + best_seg.end) / 2.0
            pairs.append((t_sub, t_audio, best_score))
            print("-----")
            print(f"Subtitle cue [{cue.start/1000.0:.2f}s]: {cue_text}")
            print(f"Best match (Score {best_score:.3f}) from ASR [{best_seg.start:.2f}-{best_seg.end:.2f}s]: {best_seg.text}")
        elif best_seg is not None:
            # even if below threshold, print what was found for inspection
            print("-----")
            print(f"(Below threshold) Subtitle cue [{cue.start/1000.0:.2f}s]: {cue_text}")
            print(f"Best match (Score {best_score:.3f}) from ASR [{best_seg.start:.2f}-{best_seg.end:.2f}s]: {best_seg.text}")

    print(f"[+] Matched {len(pairs)} subtitle–audio pairs above threshold {similarity_threshold}")
    return pairs


def fit_time_mapping(pairs: List[Tuple[float, float, float]]):
    """
    Fit t_audio ≈ a * t_sub + b using least squares.
    """
    if len(pairs) < 2:
        raise RuntimeError("Not enough pairs to fit a time mapping (need at least 2).")

    t_sub = np.array([p[0] for p in pairs])
    t_audio = np.array([p[1] for p in pairs])

    A = np.vstack([t_sub, np.ones_like(t_sub)]).T
    a, b = np.linalg.lstsq(A, t_audio, rcond=None)[0]

    # Compute residuals for info
    t_pred = a * t_sub + b
    residuals = t_audio - t_pred
    mae = np.mean(np.abs(residuals))
    rms = np.sqrt(np.mean(residuals**2))

    print(f"[+] Fitted mapping: t_audio ≈ {a:.6f} * t_sub + {b:.3f}")
    print(f"    Mean abs error: {mae:.3f} s, RMS error: {rms:.3f} s")

    return a, b


def apply_time_mapping_to_subs(
    subs: pysubs2.SSAFile,
    a: float,
    b: float,
    min_gap_ms: int = 10,
):
    """
    Apply t_new = a * t_old + b (seconds) to every cue in the subtitle file.
    """
    print("[+] Applying time mapping to all subtitle cues...")

    for line in subs:
        start_s = line.start / 1000.0
        end_s = line.end / 1000.0

        new_start_s = max(0.0, a * start_s + b)
        new_end_s = max(new_start_s + min_gap_ms / 1000.0, a * end_s + b)

        line.start = int(round(new_start_s * 1000))
        line.end = int(round(new_end_s * 1000))


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Automatically sync subtitles to a video using Whisper ASR."
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("subtitles", help="Path to subtitle file (SRT, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output subtitle path (default: input name with _synced.srt)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name (tiny, base, small, medium, large). Default: small",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Audio sample duration in seconds from start of video. Default: 300",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Text similarity threshold for matching (0–1). Default: 0.6",
    )

    args = parser.parse_args()

    video_path = args.video
    subs_path = args.subtitles
    out_path = args.output
    model_name = args.model
    duration = args.duration
    threshold = args.threshold

    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(subs_path):
        print(f"Error: subtitle file not found: {subs_path}", file=sys.stderr)
        sys.exit(1)

    if out_path is None:
        base, ext = os.path.splitext(subs_path)
        out_path = f"{base}_synced{ext or '.srt'}"

    print("[+] Loading subtitles...")
    subs = pysubs2.load(subs_path)

    # Extract subset of cues for alignment
    cues_for_alignment = get_subtitle_cues(subs, max_cues=200)
    if not cues_for_alignment:
        print("Error: no subtitle cues found to align.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "sample.wav")
        extract_audio(video_path, audio_path, start=0.0, duration=duration)

        # Run ASR
        segments = run_asr(audio_path, model_name=model_name)

        if not segments:
            print("Error: ASR produced no segments.", file=sys.stderr)
            sys.exit(1)

        # Match
        pairs = match_segments_to_cues(segments, cues_for_alignment, similarity_threshold=threshold)
        if len(pairs) < 2:
            print(
                f"Error: not enough matches ({len(pairs)}) above threshold {threshold} "
                f"to compute a reliable mapping.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Fit mapping
        a, b = fit_time_mapping(pairs)

    # Apply mapping
    apply_time_mapping_to_subs(subs, a, b)

    # Save
    subs.save(out_path)
    print(f"[+] Synced subtitles saved to: {out_path}")


if __name__ == "__main__":
    main()
