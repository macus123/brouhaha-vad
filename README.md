## Algo Overview

The script processes audio files to create balanced datasets with a 1:1 speech-to-non-speech ratio while maintaining natural audio flow.

## 1. **Audio Segmentation Algorithm**

### Step 1: Load and Analyze Audio
```python
def load_and_analyze_audio(self, input_wav: str, ground_truth: str):
    # Load audio file
    audio = AudioSegment.from_file(input_wav)
    # Read ground truth annotations (speech timestamps)
    speech_segments = read_ground_truth(ground_truth)
    # Calculate non-speech segments (gaps between speech)
    non_speech_segments = get_non_speech_segments(speech_segments, total_duration)
```

**Example:**
- Input: 60-minute audio file
- Ground truth: [(10.0, 15.0), (20.0, 30.0), (40.0, 45.0)] (speech segments)
- Non-speech segments: [(0.0, 10.0), (15.0, 20.0), (30.0, 40.0), (45.0, 60.0)]

### Step 2: Speech Padding Algorithm
```python
def _add_padding_to_speech(self, speech_segments, total_duration_sec):
    padding_sec = self.config.speech_padding_ms / 1000  # Default: 200ms
    for start, end in speech_segments:
        padded_start = max(0, start - padding_sec)
        padded_end = min(total_duration_sec, end + padding_sec)
```

**Example:**
- Original speech: (10.0, 15.0)
- With 200ms padding: (9.8, 15.2)
- This captures natural speech onset/offset

### Step 3: Segment Merging Algorithm
```python
def _merge_overlapping_segments(self, segments):
    # Sort segments by start time
    segments.sort()
    merged = []
    current_start, current_end = segments[0]
    
    for start, end in segments[1:]:
        if start <= current_end:  # Overlapping
            current_end = max(current_end, end)  # Merge
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
```

**Example:**
- Padded segments: [(9.8, 15.2), (19.8, 30.2), (39.8, 45.2)]
- If segments overlap, they merge into continuous blocks

## 2. **Silence Distribution Algo**

This algo is responsible for distributing silence throughout the audio.
Silence is reserved to prevent large continuous segments of speech and silence if either is front-loaded prematurely.
This results in a more natural flow in the final output.

### Step 1: Reserve Silence for Interspersing
```python
def _distribute_silence_intelligently(self, speech_segments, silence_segments, target_per_type_ms):
    # Reserve 40% of silence for interspersing between speech
    reserved_silence_ms = target_per_type_ms * 0.4  # 40% reserve
    primary_silence_ms = target_per_type_ms * 0.6   # 60% primary
```

**Example:**
- Target: 30 minutes total (15 min speech, 15 min silence)
- Reserved silence: 6 minutes (for interspersing)
- Primary silence: 9 minutes (for main timeline)

### Step 2: Smart Silence Selection
```python
# Sort silence by duration - shortest first for reserving
short_silence_segments = sorted(silence_segments, key=lambda x: x.duration)

# Reserve short segments for interspersing
for segment in short_silence_segments:
    if reserved_silence_duration < reserved_silence_ms:
        reserved_silence.append(segment)
```

**Logic:** Short silence segments are better for interspersing because they create natural pauses without long dead air.

### Step 3: Alternating Timeline Algorithm
```python
def _alternate_segments(self, speech_segments, silence_segments, speech_quota, silence_quota):
    # Start with some silence for natural beginning
    # Then alternate: speech -> silence -> speech -> silence
    
    while quotas_remaining and segments_available:
        if last_type_added == "speech":
            add_silence_segment()
        elif last_type_added == "non-speech":
            add_speech_segment()
```

**Example Timeline Creation:**
1. Start: 30s silence (natural intro)
2. Add: 120s speech
3. Add: 45s silence (natural break)
4. Add: 90s speech
5. Continue alternating...

## 3. **Advanced Silence Interspersing Algorithm**

### Step 1: Find Long Speech Runs
```python
def _find_speech_runs(self, segments):
    # Identify consecutive speech segments
    # A "run" is 3+ consecutive speech segments
    speech_runs = []
    current_run = []
    
    for segment in segments:
        if segment.type == "speech":
            current_run.append(segment)
        else:
            if len(current_run) >= 3:  # Long run found
                speech_runs.append(current_run)
```

**Example:**
- Segments: [speech, speech, speech, speech, silence, speech, speech]
- Identified run: segments 0-3 (4 consecutive speech segments)

### Step 2: Strategic Silence Insertion
```python
def _intersperse_reserved_silence(self, balanced_segments, reserved_silence):
    # For each long speech run, insert silence at strategic points
    for run in longest_runs_first:
        run_length = len(run)
        num_to_insert = run_length // 3  # Insert every 3rd position
        
        # Calculate insertion positions
        positions = []
        for i in range(1, num_to_insert + 1):
            pos = run_start + (i * run_length) // (num_to_insert + 1)
            positions.append(pos)
```

**Example:**
- Speech run of 9 segments: [S1, S2, S3, S4, S5, S6, S7, S8, S9]
- Insert 3 silence segments at positions 2, 5, 7
- Result: [S1, S2, silence, S3, S4, silence, S5, S6, silence, S7, S8, S9]

## 4. **Continuity Analysis Algorithm**

### Tracking Segment Continuity
```python
def compile_audio_from_segments(self, segments):
    for i, segment in enumerate(segments):
        # Check if segment is continuous with previous
        is_continuous = (i == 0 or 
                        abs(segment.start - segments[i-1].end) < 0.001)
        
        timestamp_map.append(TimestampMapping(
            is_continuous=is_continuous,
            # ... other fields
        ))
```

**Continuity Definition:**
- Continuous: Adjacent segments in original audio (gap < 1ms)
- Non-continuous: Segments with gaps between them

### Sequence Analysis
```python
def analyze_continuity(timestamps):
    sequences = []
    current_sequence = []
    
    for ts in timestamps:
        if ts.is_continuous:
            current_sequence.append(ts)
        else:
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = [ts]
```

**Example Analysis:**
- Input: [continuous, continuous, gap, continuous, continuous, continuous]
- Sequences: [[0,1], [3,4,5]] (2 sequences, lengths 2 and 3)

## 5. **Comprehensive Statistics Algorithm**

### Multi-Set Processing
```python
def process_batch(self, input_files, output_dir):
    # Create TEST/DEV/TRAIN splits
    for file in input_files:
        # Main balanced audio -> TEST set
        # Remaining audio split -> DEV (20%) and TRAIN (80%)
        
        result = process_single_file(file)
        
        # Update statistics for each set separately
        update_test_stats(result.balanced_audio)
        update_dev_stats(result.dev_audio)
        update_train_stats(result.train_audio)
```

## **Example Walkthrough**

### Input:
- Audio file: 60 minutes
- Speech content: 35 minutes (scattered throughout)
- Silence content: 25 minutes
- Target: 8 minutes balanced (4 min speech, 4 min silence)

### Processing Steps:

1. **Segment Extraction:**
   - Extract 45 speech segments (total 35 min)
   - Extract 38 silence segments (total 25 min)

2. **Padding & Merging:**
   - Add 200ms padding to speech segments
   - Merge overlapping segments
   - Result: 32 merged speech segments, 30 silence segments

3. **Silence Distribution:**
   - Reserve 1.6 min silence for interspersing
   - Use 2.4 min silence for main timeline

4. **Timeline Creation:**
   - Start: 15s silence
   - Add: 45s speech
   - Add: 20s silence
   - Add: 60s speech
   - Continue alternating until 4 min each type

5. **Interspersing:**
   - Find speech runs of 3+ segments
   - Insert reserved silence in long runs
   - Result: Natural-sounding audio flow

### Expected Output:
- **TEST set:** 8 minutes balanced audio (50% speech, 50% silence)
- **DEV set:** ~2 minutes from remaining audio
- **TRAIN set:** ~8 minutes from remaining audio
- **Statistics:** Detailed continuity analysis, sequence information

# Speech Processing Algorithm Design: Multi-file Speech/Silence Balancing

## Core Algorithm Philosophy

Let me walk you through how our temporal sequence generator works, focusing on speech segment prioritization, quota management, and timeline construction across multiple files.

## 1. Speech Prioritization Strategy

The algorithm fundamentally prioritizes speech integrity through several mechanisms:

```python
# Speech segments are never truncated, unlike silence segments
if segment.type == "speech" and speech_used < file_speech_quota[file_id]:
    if speech_used + segment.duration <= file_speech_quota[file_id]:
        file_timeline.append(segment)
        speech_used += segment.duration
    # Don't truncate speech - prefer to skip rather than truncate
```

This is a critical design choice - **we never truncate speech segments** as doing so would harm the linguistic coherence of the resulting dataset. Silence, on the other hand, can be truncated without semantic loss:

```python
# Silence can be truncated to fit quota precisely
remaining = file_silence_quota[file_id] - silence_used
if remaining > 100:  # Only add if at least 100ms
    partial = AudioSegmentInfo(
        start=segment.start,
        end=segment.start + (remaining / 1000),
        type=segment.type,
        source_file=segment.source_file,
        file_id=segment.file_id
    )
    file_timeline.append(partial)
    silence_used = file_silence_quota[file_id]
```

## 2. Multi-File Quota Management

The quota management system works with three levels of allocation:

1. **Global quota**: Total speech/silence across all files
2. **Per-file quota**: Proportional allocation to each source file 
3. **Segment selection**: Taking segments in temporal order until quota is filled

```python
# Calculate quota per file based on available content
file_speech_content = {}
file_silence_content = {}
for file_id, segments in segments_by_file.items():
    file_speech_content[file_id] = sum(s.duration for s in segments if s.type == "speech")
    file_silence_content[file_id] = sum(s.duration for s in segments if s.type == "non-speech")

# Assign quota proportionally to each file
file_speech_quota = {}
file_silence_quota = {}
for file_id in segments_by_file:
    if total_speech > 0:
        file_speech_quota[file_id] = speech_quota * (file_speech_content[file_id] / total_speech)
```

This proportional allocation ensures each source file contributes according to its content availability, preventing over-representation of any single file.

## 3. Timeline Construction and Silence Distribution

The timeline construction is handled by `_distribute_silence_intelligently_multi_file`, which employs a sophisticated approach:

```python
# Reserve silence for interspersing (a portion defined by silence_reserve_ratio)
reserved_silence_ms = silence_target_ms * self.config.silence_reserve_ratio
primary_silence_ms = silence_target_ms - reserved_silence_ms
```

The algorithm performs a two-stage silence allocation:
1. Primary silence is selected based on temporal position
2. Reserved silence is strategically interspersed between speech segments

```python
# Take some percentage of shortest silence segments for reservation
reservation_count = max(1, int(len(silence_for_reservation) * 0.3))
for i in range(min(reservation_count, len(silence_for_reservation))):
    segment = silence_for_reservation[i]
    if reserved_silence_duration < reserved_silence_ms:
        reserved_silence.append(segment)
        reserved_silence_duration += segment.duration
```

Later, this reserved silence is inserted into "speech runs" - consecutive speech segments that would benefit from silence breaks:

```python
# Find long speech runs and insert silence
speech_runs = self._find_speech_runs(balanced_segments)
speech_runs.sort(key=lambda x: x[1] - x[0], reverse=True)  # Longest first

for run_start, run_end in speech_runs:
    run_length = run_end - run_start + 1
    num_to_insert = run_length // 3  # Insert silence after every ~3 segments
    
    if num_to_insert > 0 and reserved_index < len(reserved_silence):
        # Calculate insertion positions
        positions = []
        for i in range(1, num_to_insert + 1):
            pos = run_start + (i * run_length) // (num_to_insert + 1)
            positions.append(pos + inserted)
```

## 4. Temporal Sequence Generation

The final stage involves grouping segments into natural sequences, implemented in `group_segments_into_temporal_sequences`:

```python
# Group segments by file first
file_segments = {}
for segment in segments:
    if segment.file_id not in file_segments:
        file_segments[segment.file_id] = []
    file_segments[segment.file_id].append(segment)
```

Then for each file, it:
1. Creates sequences based on temporal continuity (gaps <= 0.5s)
2. Targets minimum sequence length (3s)
3. Balances speech and silence content
4. Merges very short sequences when appropriate

```python
# If current sequence doesn't meet criteria, try to extend it despite the gap
elif current_duration < min_sequence_length or not (has_speech and has_silence):
    # If adding this segment would create a balanced sequence, include it
    if (has_speech and segment.type == "non-speech") or (has_silence and segment.type == "speech"):
        current_sequence.append(segment)
```

This creates sequences that maintain natural speech patterns while ensuring balanced representation.

## 5. Set Creation (TEST/DEV/TRAIN)

For the TEST set, we target specific duration and speech ratio. For DEV and TRAIN sets, we:

1. Find remaining segments after TEST set creation
2. Split them according to `dev_ratio` (default 20%)
3. Apply the same balancing algorithm to each set

```python
# Find remaining segments after TEST set creation
used_segments = set(id(segment) for segment in test_segments)
remaining_segments = [seg for seg in all_segments if id(seg) not in used_segments]

# Split remaining segments into DEV and TRAIN
dev_segments, train_segments = self._split_remaining_segments(remaining_segments)
```

This creates coherent sequences across all three sets while maintaining the target speech-to-silence ratio.

## Performance and Edge Cases

The algorithm handles edge cases gracefully:
- Insufficient content: Adjusts targets proportionally
- Empty files: Skips without breaking
- Very short segments: Merges with neighbors when appropriate
- Long speech runs: Injects silence to create natural breaks

The complexity is approximately O(n log n) due to sorting operations, with memory usage proportional to the total content processed.

Any questions about specific parts of the implementation?
