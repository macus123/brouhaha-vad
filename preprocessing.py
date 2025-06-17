import os
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from pathlib import Path

def read_ground_truth(gt_file):
    """Read ground truth segments from file."""
    segments = []
    with open(gt_file, 'r') as f:
        for line in f:
            start, end, _ = line.strip().split('\t', 2)
            segments.append((float(start), float(end)))
    return segments

def get_non_speech_segments(speech_segments, total_duration):
    """Get non-speech segments from speech segments."""
    non_speech = []
    current_time = 0
    
    for start, end in sorted(speech_segments):
        if current_time < start:
            non_speech.append((current_time, start))
        current_time = end
    
    if current_time < total_duration:
        non_speech.append((current_time, total_duration))
    
    return non_speech

def process_audio_file(audio_path, gt_path, output_dir):
    """Process single audio file and split into speech/non-speech segments."""
    print(f"Processing {audio_path}")
    
    # Read audio file
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio) / 1000.0  # Convert to seconds
    
    # Read ground truth segments
    speech_segments = read_ground_truth(gt_path)
    non_speech_segments = get_non_speech_segments(speech_segments, total_duration)
    
    # Create output directories
    speech_dir = os.path.join(output_dir, "speech")
    non_speech_dir = os.path.join(output_dir, "non_speech")
    os.makedirs(speech_dir, exist_ok=True)
    os.makedirs(non_speech_dir, exist_ok=True)
    
    # Extract speech segments
    for i, (start, end) in enumerate(speech_segments):
        segment = audio[int(start*1000):int(end*1000)]
        output_path = os.path.join(
            speech_dir, 
            f"{Path(audio_path).stem}_speech_{i:03d}.wav"
        )
        segment.export(output_path, format="wav")
    
    # Extract non-speech segments
    for i, (start, end) in enumerate(non_speech_segments):
        segment = audio[int(start*1000):int(end*1000)]
        output_path = os.path.join(
            non_speech_dir,
            f"{Path(audio_path).stem}_non_speech_{i:03d}.wav"
        )
        segment.export(output_path, format="wav")

def main():
    base_dir = "VAD_Input"
    datasets = ["DEV", "TEST", "TRAIN"]
    
    for dataset in datasets:
        audio_dir = os.path.join(base_dir, "Audio", dataset)
        gt_dir = os.path.join("Ground", dataset)
        output_dir = os.path.join("VAD_Output", dataset)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each audio file
        for audio_file in os.listdir(audio_dir):
            if not audio_file.endswith(('.wav', '.mp3')):
                continue
                
            audio_path = os.path.join(audio_dir, audio_file)
            gt_file = audio_file.rsplit('.', 1)[0] + '.txt'
            gt_path = os.path.join(gt_dir, gt_file)
            
            if os.path.exists(gt_path):
                process_audio_file(audio_path, gt_path, output_dir)
            else:
                print(f"Ground truth file not found for {audio_file}")

if __name__ == "__main__":
    main()