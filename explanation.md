create_sample_dataset.ipynb:
pulls and splits atco2 audio dataset into VAD_Input/Audio/{DEV, TEST, TRAIN} splits
pulls ground truths for each of the .wav files into the following format: (speech timestamp start) (speech timestamp end) (text representation of speech)

split_seg.py:
parses and matches ground truth to corresponding .wav file based on file name and runs the following operations:
1. separates speech and non-speech portions based on ground truth annotation files
2. extracts all speech segments from a .wav file, concatenating them into a single .wav file
3. extracts all non-speech segments, concatenating them into another .wav
4. notes metadata consisting of both speech and silence timestamps in .txt format
5. handles empty audio segments by inserting minimal silence (eg. <0s of silence)

main.py:
1. processes the extracted non-speech segments with brouhaha

expected outputs from extract_noise_profiles.py:
Noise_Profiles/
├── low_frequency/         # Rumbles, HVAC sounds (20Hz-200Hz)
│   ├── standard_1s_concat.wav
│   ├── standard_2s_concat.wav
│   └── standard_5s_synthetic.wav
├── mid_frequency/         # Human activity, traffic (200Hz-2kHz)
│   └── [similar files]
├── high_frequency/        # Hisses, electronic whines (2kHz+)
│   └── [similar files]
├── broadband/             # Mixed frequency noise
│   └── [similar files]
├── noise_summary.png      # Visual breakdown of noise types
└── noise_summary.txt      # Statistical analysis of noise

from create_noise_library.py:
Noise_Library/
├── standard_noise_-20dB.wav   # Very loud noise sample
├── standard_noise_-10dB.wav   # Strong noise sample
├── standard_noise_0dB.wav     # Moderate noise sample
├── standard_noise_10dB.wav    # Subtle noise sample
├── mixed_snr_noise_library.wav  # All SNR levels in one file
└── README.txt                   # Documentation

noise_merge.py:
1. Loads a speech file
2. Adds noise padding before and after the speech (length controlled by padding_ratio param)
3. Mixes noise into the speech at a controlled SNR level
4. Creates visualisations showing each component

padding_ratio param: changes how much noise padding to add, eg. 0.5 = 50% of speech length
target_snr param: controls how prominent the noise is during the speech

TODO:
create a function where it takes in a .wav file, args are (x) hours for the length of the final output
input .wav -> split portions into speech and non-speech based on ground truth -> run brouhaha on non-speech portions -> take only the top and bottom 20% portions of SNR -> extract noise from those portions (TBC) 

create a function where it takes in a .wav file, args are (x) hours for the length of the final output
input .wav -> split portions into speech and non-speech based on ground truth -> maybe generate a temp metadata file mapping split portions to timestamps of the original .wav file in order to respect the temporal order for the next operation -> centre around speech portions with a bit of padding, splice everything together including non-speech portions to achieve the (x) hours parameter eg. 10 hrs with 7 hrs of speech, x parameter is 2 hours, so must achieve a ratio of 1:1 speech to non-speech with respect to the temporal order, output would be a recompiled version of the .wav in 2 hrs, the excess speech and excess non-speech is to be compiled into 2 separate .wav files, also respecting the temporal order, so the outputs would be 1. .wav file subject to args factor, 2. remaining excess speech not used in the primary .wav file compiled into a separate .wav file, 3. remaining excess non-speech not used in the primary .wav file compiled into a separate .wav file

required function script:
1. Inputs: .wav file containing speech and non-speech segments, ground truth .txt file detailing speech and non-speech segments; Args: target length of final .wav file, desired ratio of speech:non speech

Parameters:
input_wav: Path to input audio file

Example: "VAD_Input/Audio/TRAIN/recording.wav"
ground_truth: Path to ground truth segments file (optional)

Default: Automatically looks for VAD_Input/Ground/{filename}.txt
Example: "VAD_Input/Ground/recording.txt"
target_hours: Desired duration in hours for the balanced output

Default: 1.0 (1 hour)
Example: 2.5 for a 2.5-hour output
speech_padding_ms: Milliseconds of padding around speech segments

Default: 200 (200ms)
Purpose: Creates more natural transitions between speech segments
output_dir: Directory where outputs will be saved

Default: "Recompiled_Output"

Input/Output Details
Input Sources
Audio File (specified by input_wav)

Can be any WAV file
Common location: "VAD_Input/Audio/{SPLIT}/{filename}.wav"
Ground Truth File

Format: Tab-separated values with speech segment timestamps
Each line: start_time\tend_time\toptional_text
Example: 23.45\t26.78\tSpeech segment
Default location: "VAD_Input/Ground/{filename}.txt"
Output Files
All outputs are saved in the specified output_dir (default: "Recompiled_Output"):

Balanced Audio

Filename: {input_filename}_balanced_{target_hours}h.wav
Content: Speech and non-speech in 1:1 ratio, preserving temporal order
Duration: Approximately the target duration (or less if insufficient audio)
Excess Speech

Filename: {input_filename}_excess_speech.wav
Content: Speech segments not used in the balanced output
Duration: Varies based on available content
Excess Non-Speech

Filename: {input_filename}_excess_non_speech.wav
Content: Non-speech segments not used in the balanced output
Duration: Varies based on available content
Metadata File

Filename: {input_filename}_recompile_metadata.txt
Content: Detailed processing information and statistics
Includes:
Original file details
Target parameters
Duration statistics for all outputs
Speech/non-speech percentages

2. Creates a perfectly balanced output with equal speech and silence at the target duration
3. Preserves excess content in separate files
4. Maintains temporal ordering throughout the process
5. Generates comprehensive metadata for the processing