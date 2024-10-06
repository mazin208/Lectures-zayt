# Lectures-zayt
A simple script to summarize an mp3 lecture using Openai Whisper for speech recognition and Bart (Facebook model) for summary then export the original text + the summary to a word file.

# How to use
1. install requirements

`pip install -r requirements.txt`

2. run app.py and pass it the lecture file.

`python app.py lecture.mp3`

3. it'll generate lecture.docx file
   
# Note
you can change the maximum and minimum number of generated summary using --max_length and --min_length arguments.

`python app.py lecture.mp3 --max_length 1000 --min_length 700`
