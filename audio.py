import soundfile
import librosa
import numpy as np
import sounddevice as sd
from tkinter import filedialog
from scipy.io.wavfile import write
import joblib


# Feature Extraction Function
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X)) if chroma else None

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T, axis=0)  # Reduced to 30
            delta_mfcc = np.mean(librosa.feature.delta(mfccs).T, axis=0)
            result = np.hstack((result, mfccs, delta_mfcc))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            mel_db = librosa.power_to_db(mel)  # Convert to log-mel
            result = np.hstack((result, mel_db))

        energy = np.mean(librosa.feature.rms(y=X).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
        result = np.hstack((result, energy, zcr))

    return result


# Load trained model and scaler
model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")


# Function to record and recognize emotion in real-time
def record_audio(duration=5, sample_rate=22050):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()

    # Save recorded audio
    file_name = "real_time_audio.wav"
    write(file_name, sample_rate, np.int16(audio_data * 32767))  # Convert float to int16
    
    return predict_emotion(file_name)


# Function to predict emotion
def predict_emotion(file_name):
    features = extract_feature(file_name)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]


# Function to upload audio file
def upload_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        return predict_emotion(file_path)


