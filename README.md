## Algo Overview

The script processes audio files to create datasets with a configurable ratio of speech-to-non-speech, prioritizing natural language flow.

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
    padding_sec = self.config.speech_padding_ms / 1000
    
    for start, end in speech_segments:
        padded_start = max(0, start - padding_sec)
        padded_end = min(total_duration_sec, end + padding_sec)
        padded_segments.append((padded_start, padded_end))
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
This function sorts segments by start time and merges any segments where start <= previous_end

**Example:**
- Padded segments: [(9.8, 15.2), (19.8, 30.2), (39.8, 45.2)]
- If segments overlap, they merge into continuous blocks

## 2. **Timeline Construction**
```python
def _create_natural_timeline(self, speech_segments, silence_segments, speech_quota, silence_quota, reserved_silence):
    # 1. Group segments by file_id
    # 2. Calculate per-file quotas proportionally
    # 3. Process each file in temporal order
    # 4. Select segments until quotas are met
    # 5. Intersperse reserved silence in long speech runs
```
The timeline construction prioritizes the following:
- Speech segments are never truncated (preserving linguistic integrity)
- Silence segments can be truncated to fit quota precisely
- Files contribute proportionally to their content availability
- Temporal ordering is preserved within each file

## 3. **Temporal Sequence Generation**

```python
def group_segments_into_temporal_sequences(self, segments):
    # Group segments by file_id
    # For each file:
    #   1. Sort segments by time
    #   2. Find natural sequences with gaps <= 0.5s
    #   3. Require sequences to have minimum length and speech+silence
    #   4. Handle orphaned segments to prevent data loss
```

The script prioritizes the following:
- Segments with gaps ≤ 0.5s are grouped together
- Sequences should be at least 3 seconds long
- Ideally contain both speech and silence for balance
- Segments can be grouped despite larger gaps if needed to meet criteria

## 4. **TEST/DEV/TRAIN Split Generation**
```python
def process_unified_output(self, input_files, test_dir, dev_dir, train_dir):
    # 1. Create balanced TEST set using target durations
    # 2. Identify segments not used in TEST set
    # 3. Split remaining content using dev_ratio (default 20%)
    # 4. Apply balancing algorithm to DEV/TRAIN separately
    # 5. Save all splits as temporal sequences
```

## 5. **Input Format**
The script expects the following directory structure:
```
VAD_Input/
  ├── Audio/
  │   ├── file1.wav
  │   └── file2.wav
  └── Ground/
      ├── file1.txt
      └── file2.txt
```

Ground truth files must be tab-separated with the following format:
```
start_time    end_time    text_content
95.003469     98.395344   Speech segment 1 content
105.330969    109.667844  Speech segment 2 content
```

# Example Walkthrough for convert_wav.py

## Input Scenario
- **Audio file**: paris_walk.wav (60 minutes)
- **Speech content**: 15 minutes (scattered throughout timeline)
- **Silence content**: 45 minutes
- **Target**: Create balanced TEST/DEV/TRAIN splits with 1:1 speech-to-silence ratio

## Processing Steps

### 1. File Loading & Analysis
```
Loading file 1/1: input_data\audio\paris_walk.wav
Total content available: 1:00:00
  Speech: 0:15:00
  Silence: 0:45:00
```

- Load audio using pydub (`AudioSegment.from_file`)
- Parse ground truth file with text content
- Add 200ms padding to all speech segments
- Merge overlapping segments after padding
- Extract silence segments from gaps between speech

### 2. Segment Analysis & Allocation
```
Multi-file speech content: 0:15:00
Multi-file non-speech content: 0:45:00
Target TEST set: 0:12:00 (0:06:00 speech, 0:06:00 silence)
```

- **Speech segments extracted**: 28 segments (15 minutes total)
- **Silence segments extracted**: 29 segments (45 minutes total)
- Calculate TEST set quota (default: 6 min speech, 6 min silence)
- Reserve 40% of silence (2:24) for interspersing in speech runs
- Remaining silence (3:36) used for primary timeline

### 3. Natural Timeline Construction
```python
# Timeline construction
speech_used = 0
silence_used = 0
last_type = None

# Process segments in temporal order
for segment in sorted_segments:
    if speech_used < speech_quota and segment.type == "speech":
        timeline.append(segment)
        speech_used += segment.duration
        last_type = "speech"
    elif silence_used < silence_quota and segment.type == "silence":
        timeline.append(segment)
        silence_used += segment.duration
        last_type = "silence"
```

The algorithm:
1. Processes segments in temporal order
2. Adds speech segments until speech quota is reached
3. Adds silence segments until silence quota is reached
4. Creates a balanced timeline with alternating speech/silence

### 4. Speech Run Analysis & Silence Interspersing
```
Finding speech runs...
Found 5 speech runs, inserting reserved silence
```

Note: Reserved Silence only applicable when a minimum amount of silence is required as configured in the parameters

1. Identify continuous runs of 3+ speech segments
2. Sort runs by length (longest first)
3. Calculate optimal insertion points in each run
4. Insert reserved silence to break up long speech runs
   - For a run of 6 speech segments: insert 2 silence segments
   - For a run of 3 speech segments: insert 1 silence segment

### 5. Temporal Sequence Creation
```
Creating TEST temporal sequences...
```

1. Identify natural sequence boundaries (gaps > 0.5s)
2. Group segments into coherent temporal sequences
3. Ensure each sequence has:
   - Minimum 3 seconds duration
   - At least 3 segments (when possible)
   - Both speech and silence (when possible)
4. Name sequences incrementally: `TEST_paris_walk_001.wav`, `TEST_paris_walk_002.wav`, etc.

Example sequences:
```
Created TEST_paris_walk_001.wav: 0:02:38 (speech: 0:00:45, silence: 0:01:53, 28.5% speech)
Created TEST_paris_walk_002.wav: 0:01:14 (speech: 0:00:32, silence: 0:00:42, 43.2% speech)
```

### 6. TEST/DEV/TRAIN Split Generation
```
TEST set created: 0:12:00 across 14 sequences
  DEV: 96 segments
  TRAIN: 381 segments
Creating DEV temporal sequences...
```

1. After creating TEST set, identify unused segments
2. Allocate 20% of remaining segments to DEV (configurable)
3. Allocate 80% of remaining segments to TRAIN
4. Apply similar temporal sequence creation to each split
5. Generate ground truth files with original speech text

Example output:
```
DEV set created: 0:09:36 across 11 sequences
Creating TRAIN temporal sequences...
TRAIN set created: 0:38:24 across 22 sequences
```

### 7. Ground Truth Generation & Text Preservation
```
# Ground truth file: TEST_paris_walk_001.txt
10.330969    14.667844    Speech segment 2 dwasdwa a da wd
148.395969   152.243469   Speech segment 3 wda awdas asd
```

1. For each output audio file, create matching ground truth file
2. Preserve original speech text content from source file
3. Adjust timestamps to match new audio file timing
4. Format as tab-separated values

### 8. Statistics Generation
```
Batch processing summary saved to Recompiled_Output\batch_processing_summary.csv
  Total: 47 sequence files recorded in CSV
```

1. Calculate detailed statistics for each output file:
   - Duration, speech percentage, sequence count
   - Source file mapping, segment continuity
2. Generate comprehensive CSV summary
3. Display processing results in terminal

## Final Output

### Directory Structure
```
Recompiled_Output/
  ├── TEST/                                # 14 sequences, 12:00 total
  │   ├── TEST_paris_walk_001.wav          # 2:38 (28.5% speech)
  │   ├── TEST_paris_walk_001.txt          # With original speech text
  │   ├── TEST_paris_walk_002.wav          # 1:14 (43.2% speech)
  │   └── ...
  ├── DEV/                                 # 11 sequences, 9:36 total
  │   └── ...
  ├── TRAIN/                               # 22 sequences, 38:24 total
  │   └── ...
  └── batch_processing_summary.csv         # Detailed statistics
```

### Output Statistics
- **TEST set**: 6:00 speech, 6:00 silence (balanced 1:1 ratio)
- **DEV set**: ~4:48 speech, ~4:48 silence (balanced 1:1 ratio)
- **TRAIN set**: ~19:12 speech, ~19:12 silence (balanced 1:1 ratio)
- **Total processed**: 60:00 (original file duration)

#### Output Structure
```
Recompiled_Output/
  ├── TEST/
  │   ├── TEST_original_filename_001.wav
  │   └── TEST_original_filename_001.txt
  ├── DEV/
  │   ├── DEV_original_filename_001.wav
  │   └── DEV_original_filename_001.txt
  ├── TRAIN/
  │   ├── TRAIN_original_filename_001.wav
  │   └── TRAIN_original_filename_001.txt
  └── batch_processing_summary.csv
```


## 2. Quota Management

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

This proportional allocation ensures each source file contributes according to its content availability, preventing over-representation of any single file. However, the proportional allocation algorithm is disregarded in favor of a simpler greedy algorithm for cases where there are many tiny files (<15s each) as each file may not accurately represent the distribution needed for the proportional quota allocated to it.

i.e. 100 files of <15s each, with only 20 files containing speech. A proportional quota would give each file an equal amount of speech quota, misrepresenting the dataset and failing to meet the output requirements as the speech files have too little quota to include in the final output.
