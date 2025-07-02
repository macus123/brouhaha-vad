import csv
import datetime
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

from pydub import AudioSegment

from split_seg import read_ground_truth, get_non_speech_segments, save_audio_safely


# --- Data Structures ---
@dataclass
class SplitStats:
    """Holds the statistics for a single data split (e.g., TEST, DEV, TRAIN)."""
    duration_h: float = 0.0
    speech_h: float = 0.0
    total_segments: int = 0
    continuous_segments: int = 0
    num_sequences: int = 0
    longest_sequence: int = 0

    @property
    def silence_h(self) -> float:
        return self.duration_h - self.speech_h

    @property
    def speech_percentage(self) -> float:
        return (self.speech_h / self.duration_h) * 100 if self.duration_h > 0 else 0


# --- Helper Functions ---

def scan_input_directory(input_dir: str = "input_data") -> List[Dict[str, str]]:
    """Scan input_data directory structure to find audio files and match with ground truth."""
    base_dir = Path(input_dir)
    audio_dir = base_dir / "audio"
    ground_truth_dir = base_dir / "ground_truth"

    if not audio_dir.exists() or not ground_truth_dir.exists():
        print(f"Error: 'audio' or 'ground_truth' directory not found in {input_dir}")
        return []

    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    matched_files = []
    for audio_file in audio_files:
        gt_file = ground_truth_dir / f"{audio_file.stem}.txt"
        if gt_file.exists():
            matched_files.append({"audio_path": str(audio_file), "ground_truth_path": str(gt_file)})
        else:
            print(f"Warning: No ground truth found for {audio_file}")

    print(f"Matched {len(matched_files)} files with ground truth")
    return matched_files


def _prepare_initial_timeline(
    input_wav_path: str, ground_truth_path: str, speech_padding_ms: int
) -> Tuple[AudioSegment, List[Dict[str, Any]]]:
    """Loads audio, processes ground truth, and creates a master timeline of all segments."""
    audio = AudioSegment.from_file(input_wav_path)
    total_duration_sec = len(audio) / 1000

    speech_segments = read_ground_truth(ground_truth_path)

    padded_speech_segments = []
    if speech_segments:
        for start, end in speech_segments:
            padded_start = max(0, start - speech_padding_ms / 1000)
            padded_end = min(total_duration_sec, end + speech_padding_ms / 1000)
            padded_speech_segments.append((padded_start, padded_end))

        padded_speech_segments.sort()
        merged_speech_segments = [padded_speech_segments[0]]
        for start, end in padded_speech_segments[1:]:
            last_start, last_end = merged_speech_segments[-1]
            if start <= last_end:
                merged_speech_segments[-1] = (last_start, max(last_end, end))
            else:
                merged_speech_segments.append((start, end))
    else:
        merged_speech_segments = []

    non_speech_segments = get_non_speech_segments(merged_speech_segments, total_duration_sec)
    timeline = []
    for start, end in merged_speech_segments:
        timeline.append({"start": start, "end": end, "type": "speech", "duration": (end - start) * 1000})
    for start, end in non_speech_segments:
        timeline.append({"start": start, "end": end, "type": "non-speech", "duration": (end - start) * 1000})

    timeline.sort(key=lambda x: x["start"])
    return audio, timeline


def _select_balanced_segments(
    timeline: List[Dict], target_per_type_ms: int, silence_reserve_ratio: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Selects segments using the full, sophisticated algorithm from the original script.
    """
    all_speech = sorted([s for s in timeline if s["type"] == "speech"], key=lambda x: x["start"])
    all_silence = [s for s in timeline if s["type"] == "non-speech"]

    # --- 1. Reserve silence with flexible quotas ---
    reserved_silence_ms = target_per_type_ms * silence_reserve_ratio
    reserved_silence, reserved_duration = [], 0
    for segment in sorted(all_silence, key=lambda x: x["duration"]):
        if reserved_duration >= reserved_silence_ms: break
        if reserved_duration + segment["duration"] <= reserved_silence_ms * 1.1:
            reserved_silence.append(segment); reserved_duration += segment["duration"]
        elif reserved_duration < reserved_silence_ms * 0.9:
            needed = reserved_silence_ms - reserved_duration
            if segment["duration"] > needed * 2:
                part1 = segment.copy()
                part1["end"] = segment["start"] + needed / 1000
                part1["duration"] = needed
                reserved_silence.append(part1)
                reserved_duration += needed
            else:
                reserved_silence.append(segment)
                reserved_duration += segment["duration"]

    reserved_ids = {id(s) for s in reserved_silence}
    primary_silence_pool = sorted([s for s in all_silence if id(s) not in reserved_ids], key=lambda x: x["start"])

    # --- 2. Build base timeline with preference-based alternation ---
    balanced_segments = []
    speech_quota = target_per_type_ms
    silence_quota = target_per_type_ms - reserved_duration
    speech_idx, silence_idx = 0, 0
    last_type_added = None

    if primary_silence_pool and silence_quota > 0: # Natural start
        segment = primary_silence_pool.pop(0)
        if segment['duration'] > silence_quota:
            segment['end'] = segment['start'] + silence_quota / 1000; segment['duration'] = silence_quota
        balanced_segments.append(segment); silence_quota -= segment['duration']; last_type_added = "non-speech"

    while (speech_quota > 0 and speech_idx < len(all_speech)) or \
          (silence_quota > 0 and silence_idx < len(primary_silence_pool)):
        add_speech = False
        can_add_speech = speech_quota > 0 and speech_idx < len(all_speech)
        can_add_silence = silence_quota > 0 and silence_idx < len(primary_silence_pool)

        if last_type_added == "speech" and can_add_silence: add_speech = False
        elif last_type_added != "speech" and can_add_speech: add_speech = True
        elif can_add_speech: add_speech = True
        elif can_add_silence: add_speech = False
        else: break
        
        if add_speech:
            segment = all_speech[speech_idx]; speech_idx += 1
            if segment['duration'] > speech_quota:
                segment['end'] = segment['start'] + speech_quota/1000; segment['duration'] = speech_quota
            balanced_segments.append(segment); speech_quota -= segment['duration']; last_type_added = "speech"
        else:
            segment = primary_silence_pool[silence_idx]; silence_idx += 1
            if segment['duration'] > silence_quota:
                segment['end'] = segment['start'] + silence_quota/1000; segment['duration'] = silence_quota
            balanced_segments.append(segment); silence_quota -= segment['duration']; last_type_added = "non-speech"

    # --- 3. Globally analyze and intersperse silence in longest speech runs first ---
    balanced_segments.sort(key=lambda x: x["start"])
    speech_runs, current_run_start = [], None
    for i, segment in enumerate(balanced_segments):
        if segment['type'] == 'speech':
            if current_run_start is None: current_run_start = i
        elif current_run_start is not None:
            if i - current_run_start >= 3: speech_runs.append((current_run_start, i-1))
            current_run_start = None
    if current_run_start is not None and len(balanced_segments) - current_run_start >= 3:
        speech_runs.append((current_run_start, len(balanced_segments)-1))

    speech_runs.sort(key=lambda r: r[1] - r[0], reverse=True)

    final_segments = balanced_segments.copy()
    inserted_count = 0
    for run_start, run_end in speech_runs:
        if not reserved_silence: break
        run_len = run_end - run_start + 1
        num_to_insert = run_len // 3
        insertion_points = [run_start + (i * run_len // (num_to_insert + 1)) for i in range(1, num_to_insert + 1)]
        for pos in sorted(insertion_points, reverse=True):
            if not reserved_silence: break
            final_segments.insert(pos + inserted_count, reserved_silence.pop(0))
    
    if reserved_silence: final_segments.extend(reserved_silence)
    final_segments.sort(key=lambda x: x["start"])

    used_ids = {id(s) for s in final_segments}
    remaining = [s for s in timeline if id(s) not in used_ids]
    return final_segments, remaining


def _split_remaining_segments(remaining: List[Dict], dev_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    """Splits remaining segments into DEV and TRAIN sets based on duration quotas."""
    if not remaining: return [], []
    total_rem_speech = sum(s['duration'] for s in remaining if s['type'] == 'speech')
    dev_speech_quota = total_rem_speech * dev_ratio
    total_rem_silence = sum(s['duration'] for s in remaining if s['type'] == 'non-speech')
    dev_silence_quota = total_rem_silence * dev_ratio
    dev_segments, train_segments = [], []
    for segment in remaining:
        if segment['type'] == 'speech' and dev_speech_quota > 0:
            dev_segments.append(segment); dev_speech_quota -= segment['duration']
        elif segment['type'] == 'non-speech' and dev_silence_quota > 0:
            dev_segments.append(segment); dev_silence_quota -= segment['duration']
        else: train_segments.append(segment)
    return dev_segments, train_segments


def _compile_audio_from_segments(segments: List[Dict], source: AudioSegment) -> Tuple[AudioSegment, List[Dict]]:
    """Compiles an AudioSegment and tracks continuity."""
    audio, timestamps = AudioSegment.empty(), []
    for i, seg in enumerate(segments):
        start_ms, end_ms = int(seg["start"] * 1000), int(seg["end"] * 1000)
        seg_audio = source[start_ms:end_ms]
        timestamps.append({
            "duration_sec": len(seg_audio) / 1000, "type": seg["type"],
            "is_continuous": i == 0 or abs(seg["start"] - segments[i - 1]["end"]) < 0.001
        })
        audio += seg_audio
    return audio, timestamps


def analyze_continuity(timestamps: List[Dict]) -> Dict:
    """Analyzes continuity to find sequences and other stats."""
    if not timestamps: return {"num_sequences": 0, "longest_sequence": 0, "total_segments": 0, "continuous_segments": 0}
    seqs, cur_len, cont_count = [], 0, 0
    for i, ts in enumerate(timestamps):
        is_cont = ts.get("is_continuous", i == 0)
        if is_cont: cur_len += 1; cont_count += 1
        else:
            if cur_len > 0: seqs.append(cur_len)
            cur_len = 1
    if cur_len > 0: seqs.append(cur_len)
    return {"num_sequences": len(seqs), "longest_sequence": max(seqs) if seqs else 0,
            "total_segments": len(timestamps), "continuous_segments": cont_count}


def recompile_audio_file(
        input_wav: str, ground_truth: str, target_hours: float, speech_padding_ms: int,
        output_dir: str, create_splits: bool, dev_ratio: float, silence_reserve_ratio: float
) -> Tuple[Dict[str, Any], FileProcessingResult]:
    """Orchestrates the processing for a single audio file."""
    file_stem = Path(input_wav).stem
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Processing file: {input_wav}")
    source_audio, timeline = _prepare_initial_timeline(input_wav, ground_truth, speech_padding_ms)
    
    target_per_type_ms = int(target_hours * 3600 * 1000) // 2
    total_speech_ms = sum(s['duration'] for s in timeline if s['type'] == 'speech')
    total_non_speech_ms = sum(s['duration'] for s in timeline if s['type'] == 'non-speech')

    if total_speech_ms < target_per_type_ms or total_non_speech_ms < target_per_type_ms:
        print("Warning: Insufficient audio. Adjusting target...")
        target_per_type_ms = min(total_speech_ms, total_non_speech_ms)

    balanced_segments, remaining = _select_balanced_segments(timeline, target_per_type_ms, silence_reserve_ratio)

    test_audio, test_ts = _compile_audio_from_segments(balanced_segments, source_audio)
    test_path = output_path / f"{file_stem}_balanced_{target_hours:.1f}h.wav"
    save_audio_safely(test_audio, test_path)

    outputs = {"TEST": (test_audio, test_ts, test_path)}
    file_result = FileProcessingResult(filename=Path(input_wav).name, original_duration_h=len(source_audio)/3600000)

    if create_splits and remaining:
        dev_segs, train_segs = _split_remaining_segments(remaining, dev_ratio)
        if dev_segs:
            dev_audio, dev_ts = _compile_audio_from_segments(dev_segs, source_audio)
            if len(dev_audio) > 0:
                outputs["DEV"] = (dev_audio, dev_ts, output_path / f"{file_stem}_dev.wav")
                save_audio_safely(dev_audio, outputs["DEV"][2])
        if train_segs:
            train_audio, train_ts = _compile_audio_from_segments(train_segs, source_audio)
            if len(train_audio) > 0:
                outputs["TRAIN"] = (train_audio, train_ts, output_path / f"{file_stem}_train.wav")
                save_audio_safely(train_audio, outputs["TRAIN"][2])

    for name, (audio, ts, _) in outputs.items():
        stats = analyze_continuity(ts)
        file_result.splits[name] = SplitStats(
            duration_h=len(audio)/3600000,
            speech_h=sum(t['duration_sec'] for t in ts if t['type'] == 'speech') / 3600, **stats)
    return outputs, file_result


def batch_recompile_audio(
        input_files: List[dict], output_dir: str, target_hours: float,
        speech_padding_ms: int, create_splits: bool, dev_ratio: float, silence_reserve_ratio: float
):
    """Processes a batch of files and generates a summary CSV."""
    out_path = Path(output_dir)
    dir_map = {"TEST": out_path / "TEST", "DEV": out_path / "DEV", "TRAIN": out_path / "TRAIN"}
    for p in dir_map.values(): p.mkdir(exist_ok=True, parents=True)

    all_results = []
    for i, file_info in enumerate(input_files):
        print(f"\n[{i + 1}/{len(input_files)}] Processing {file_info['audio_path']}...")
        try:
            outputs, file_result = recompile_audio_file(
                input_wav=file_info["audio_path"], ground_truth=file_info["ground_truth_path"],
                target_hours=target_hours, speech_padding_ms=speech_padding_ms,
                output_dir=str(dir_map["TEST"]), create_splits=create_splits,
                dev_ratio=dev_ratio, silence_reserve_ratio=silence_reserve_ratio
            )
            all_results.append(file_result)
            for name, (_, _, temp_path) in outputs.items():
                if name in dir_map and temp_path.exists():
                    final_path = dir_map[name] / temp_path.name
                    if temp_path.resolve() != final_path.resolve():
                        shutil.move(str(temp_path), str(final_path))
                    print(f"    - {name} output: {final_path.name}")
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append(FileProcessingResult(filename=Path(file_info['audio_path']).name, error=str(e)))
    
    create_batch_summary_csv(all_results, output_dir)


def _write_csv_split_row(writer, filename, original_h, split_name, stats: SplitStats):
    """Helper to write a single row for a split into the CSV."""
    writer.writerow({
        "Filename": filename, "SplitType": split_name, "OriginalDuration(h)": f"{original_h:.4f}",
        "OutputDuration(h)": f"{stats.duration_h:.4f}", "SpeechDuration(h)": f"{stats.speech_h:.4f}",
        "SpeechPercentage": f"{stats.speech_percentage:.2f}", "SilenceDuration(h)": f"{stats.silence_h:.4f}",
        "SilencePercentage": f"{100-stats.speech_percentage:.2f}", "TotalSegments": stats.total_segments,
        "ContinuousSegments": stats.continuous_segments, "Sequences": stats.num_sequences,
        "LongestSequence": stats.longest_sequence
    })


def create_batch_summary_csv(results: List[FileProcessingResult], output_dir: str):
    """Creates a CSV summary file for the entire batch processing job."""
    summary_file = Path(output_dir) / "batch_processing_summary.csv"
    headers = ["Filename", "SplitType", "OriginalDuration(h)", "OutputDuration(h)", "SpeechDuration(h)",
               "SpeechPercentage", "SilenceDuration(h)", "SilencePercentage", "TotalSegments",
               "ContinuousSegments", "Sequences", "LongestSequence"]
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        agg_stats, total_orig_h = SplitStats(), 0
        for res in results:
            if res.error:
                writer.writerow({"Filename": res.filename, "SplitType": "ERROR", "OriginalDuration(h)": res.error})
                continue
            total_orig_h += res.original_duration_h
            for name, stats in res.splits.items():
                _write_csv_split_row(writer, res.filename, res.original_duration_h, name, stats)
                agg_stats.duration_h += stats.duration_h; agg_stats.speech_h += stats.speech_h
                agg_stats.total_segments += stats.total_segments; agg_stats.continuous_segments += stats.continuous_segments
                agg_stats.num_sequences += stats.num_sequences
                agg_stats.longest_sequence = max(agg_stats.longest_sequence, stats.longest_sequence)
        writer.writerow({})
        _write_csv_split_row(writer, "AGGREGATE_SUMMARY", total_orig_h, "ALL", agg_stats)
    print(f"\nBatch processing summary saved to {summary_file}")


def main():
    """High-level wrapper to process all audio files in a directory."""
    input_files = scan_input_directory(input_dir="input_data")
    if not input_files: return
    batch_recompile_audio(
        input_files=input_files, output_dir="Recompiled_Output",
        target_hours=0.135, speech_padding_ms=200, create_splits=True,
        dev_ratio=0.2, silence_reserve_ratio=0.4
    )

if __name__ == "__main__":
    main()