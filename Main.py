
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio

import librosa
import numpy as np
import tensorflow as tf
import pyaudio

# Load saved model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define emotions
emotions = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'ps'
}

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define callback function to read audio input
def callback(in_data, frame_count, time_info, status):
    # Convert input data to float32 numpy array
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    
    # Resample audio to 22050 Hz (required by model)
    audio_data = librosa.resample(audio_data, 44100, 22050)
    
    # Compute Mel spectrogram features
    mel_spec = librosa.feature.melspectrogram(audio_data, sr=22050, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    
    # Make prediction with model
    predictions = model.predict(mel_spec_db)[0]
    predicted_emotion = emotions[np.argmax(predictions)]
    print(predicted_emotion)
    
    return (in_data, pyaudio.paContinue)

# Open audio stream with callback function
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024,
                stream_callback=callback)

# Start audio stream
stream.start_stream()

# Wait for stream to finish
while stream.is_active():
    pass

# Stop audio stream
stream.stop_stream()
stream.close()

# Terminate PyAudio 
p.terminate()

lists = []
