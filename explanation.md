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

expected ground truth format:
start_time[TAB]end_time[TAB]optional_text

eg.
13.450000	15.320000	Speech segment 1
17.890000	22.670000	Speech segment 2

TESTING

original 40 min video:
Found 150 speech segments
Speech: 388.26s (16.0% of audio)
Non-speech: 2041.55s (84.0% of audio)
Audio file: my_gt_data\audio\recording.wav
Ground truth file: my_gt_data\ground_truth\recording.txt
Found 150 speech segments
Speech duration: 388.26s
Non-speech duration: 2041.55s
Total duration: 2429.81s

coverted using convert_wav.py to 8 mins, achieving a 1:1 ratio of speech to non-speech
Target: 0.14 hours (486.0 seconds) with 1:1 speech/non-speech ratio
Original audio: 0:40:29.806000
Balanced output: my_gt_data\Recompiled_Output\recording_balanced_0.1h.wav
Speech content: 0.07 hours
Non-speech content: 0.07 hours
Excess speech: 0.06 hours
Excess non-speech: 0.48 hours

testing using brouhaha for the proportion of speech to non-speech segments in the new spliced .wav file:
Speech duration: 200.00s
Non-speech duration: 286.00s
Total duration: 486.00s
Speech: 200.00s (41.2% of audio)
Non-speech: 286.00s (58.8% of audio)

spliced audio has an increase in speech proportion of 16% to 41.2%, upon re-evaluation using the brouhaha model

TLDR:
Pipeline attempts to equalize ratio of speech to non-speech based on an arg (x hours), achieved the following results:
Speech: 388.26s (16.0% of audio)
Non-speech: 2041.55s (84.0% of audio)

new:
Speech: 200.00s (41.2% of audio)
Non-speech: 286.00s (58.8% of audio)

TODO:
ideally there will be no leftover audio
TEST set would consist of primary recompiled audio according to 1:1 ratio, specified length of time
DEV and TRAIN set to be split 20/80, using the leftover unused sounds and silence
importance to context, ideally use the full speech segment with padding as much as possible, do not cut it off if necessary
implement quotas for speech and silence segments, eg. 10mins quota for both, parse the length of the .wav and pick up complete speech segments to hit the quota, using an algorithm, ensure that the speech segments and its neighbouring silence segments both add up to the quota as much as possible, eg. if speech segment 1 has too long of a silence segment right after which would not fit into the quota, skip that and search for the next speech segment that has an appropriate silence segment. if this is not possible, start over again and start looking for opportunities to trim the neighbouring silence segments to fit within the quota. keep metadata of timestamps trimmed

keep metadata of the distribution of timestamps, for users to confirm that the ratio has been achieved according to the timestamps, repeat this metadata generation for train and dev sets using the leftover audio, adhere to temporal order in all cases

tell users distribution of segments in recompiled .wav


1. separate speech and non speech portions from given .wav based on ground truth timestamps -> 1 .wav splitting into 2, 1 speech and 1 non-speech
2. run brouhaha model on the non-speech files so that we can find the top and bottom 20th percentile of SNR ratio
3. bottom 20%, extract noise from .wav in this category
4. create sound library with all the noises extracted
5. handover to zaiming