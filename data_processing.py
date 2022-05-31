import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks

def extract_features(filename):

    features = pd.DataFrame()

    # get the filename
    features = features.append({"filename": os.path.basename(filename)}, ignore_index=True)

    #get length of audio file
    duration = librosa.get_duration(filename=filename)
    sample_rate = librosa.get_samplerate(filename)
    features['length'] = int(duration * sample_rate)

    # load the audio file
    audio_data, sample_rate = librosa.load(filename, mono=True, sr=sample_rate)

    # get chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)

    # get rms
    rms = librosa.feature.rms(y=audio_data)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # get spectral_centroid
    spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_var'] = np.var(spec_cent)

    # get spectral_bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_var'] = np.var(spec_bw)

    # get spectral_rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    # get zero_crossing_rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    # get harmonic effect
    harmonic = librosa.effects.harmonic(audio_data)
    features['harmonic_mean'] = np.mean(harmonic)
    features['harmonic_var'] = np.var(harmonic)

    # get tempo
    tempo = librosa.beat.tempo(audio_data)
    features['tempo'] = np.mean(tempo)

    # get mfcc
    mfccs = librosa.feature.mfcc(audio_data, sample_rate, n_mfcc=20)
    for i, mfcc in enumerate(mfccs):
        mfcc = mfcc.T
        mfcc_mean = "mfcc{0}".format(i+1) + "_mean"
        mfcc_var  = "mfcc{0}".format(i+1) + "_var"

        features[mfcc_mean] = np.mean(mfcc)
        features[mfcc_var] = np.var(mfcc)

    # extract label from file name
    features['label'] = os.path.basename(filename).split('.')[0]

    return features

def extract_audio_chunks(audio_file, seconds_per_chunk = 3000, temp_dir = 'temp'):

    file_list = []

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    filename = os.path.basename(audio_file)

    temp_audio = AudioSegment.from_file(audio_file , "wav")
    audio_chunks = make_chunks(temp_audio, seconds_per_chunk) 

    #remove the silent last chunk
    del audio_chunks[-1]

    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(audio_chunks):
        chunk_name = os.path.join(temp_dir, os.path.splitext(filename)[0] + ".{0}.wav".format(i))
        chunk.export(chunk_name, format="wav")
        file_list.append(chunk_name)

    return file_list
