# Lectures-zayt
A simple script to summarize an mp3 lecture using [Openai Whisper](https://github.com/openai/whisper) model for audio transcription and the [BART](https://huggingface.co/facebook/bart-large-cnn) model for text summarization .
The script transcribes an audio file, generates a summary of the transcription, and saves both the transcription and summary to a Word document.

## Requirements

- Python 3.7 or higher

To install the required libraries, you can run:


```pip install whisper transformers torch python-docx```

## Usage
To run the script, use the following command:
```python final.py <audio_file> [--max_length MAX_LENGTH] [--min_length MIN_LENGTH]```

## Arguments
audio_file (required): Path to the audio file (e.g., lecture.mp3).
--max_length (optional): The maximum length of the summary. Default is 500.
--min_length (optional): The minimum length of the summary. Default is 200.
## Example
```python final.py lecture.mp3 --max_length 400 --min_length 150```

## Notes
* The script uses default values of 500 for max_length and 200 for min_length if they are not provided.
* because of the limit of 1024 chunk size the code splits the transcription into 2 parts then do the summarization.
