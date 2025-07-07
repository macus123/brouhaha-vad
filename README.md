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


