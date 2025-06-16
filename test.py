from brouhaha_pipeline import process_audio
from helpers import analyze_speech_silence_ratio, extract_speech_silence_segments, create_representative_audio_bucketed
import librosa

data_path = "data/paris_walk.wav"
# data_path = "results/recompiled_audio/output.wav"
output_dir = "./results"

# brouhaha processing to detect speech segments, estimate SNR and C50
results = process_audio(
    audio_paths="data/paris_walk.wav",
    out_dir="./results"
)

# 2. Calculate speech-silence ratio
audio_file = data_path
y, sr = librosa.load(audio_file, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

ratio_analysis = analyze_speech_silence_ratio(
    "results/rttm_files/paris_walk.rttm", 
    duration
)
print(f"Speech to silence ratio: {ratio_analysis['ratio']:.2f}")
print(f"Speech: {ratio_analysis['speech_percentage']:.1f}% of total audio")

# 3. Extract speech and silence segments
segments = extract_speech_silence_segments(
    audio_file, 
    "results/rttm_files/paris_walk.rttm", 
    "extracted_segments"
)

# 4. Create balanced audio (1:1 ratio)
balanced = create_representative_audio_bucketed(
    segments['speech_files'],
    segments['silence_files'],
    "results/recompiled_audio/output.wav",
    target_ratio=1.0
)
print(f"\nRecompiled audio:")
print(f"Target ratio: {balanced['target_ratio']}")
print(f"Achieved ratio: {balanced['achieved_ratio']:.2f}")
print(f"Total duration: {balanced['total_duration']:.1f}s")
print(f"Speech duration: {balanced['speech_duration']:.1f}s")
print(f"Silence duration: {balanced['silence_duration']:.1f}s")