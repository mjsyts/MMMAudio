import librosa
import numpy as np
import matplotlib.pyplot as plt

def mel_to_hz_librosa_results():
    mels = [100.0 * (i + 1) for i in range(8)]
    hz_librosa_results = [float(librosa.mel_to_hz(mel)) for mel in mels]

    print("Librosa mel to hz results:")
    print("mels: ", mels)
    print("hz_librosa_results: ", hz_librosa_results)

def np_linspace_results():
    start = 0.0
    stop = 1.0
    num = 8
    linspace_results = np.linspace(start, stop, num).tolist()

    print("NumPy linspace results:")
    print("start: ", start)
    print("stop: ", stop)
    print("num: ", num)
    print("linspace_results: ", linspace_results)
    
def mel_frequencies_results():
    n_mels = 32
    fmin = 20.0
    fmax = 20000.0
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax).tolist()

    print("Mel frequencies results:")
    print("n_mels: ", n_mels)
    print("fmin: ", fmin)
    print("fmax: ", fmax)
    print("mel_frequencies: ", mel_frequencies)

def fft_frequencies_results():
    n_fft = 512
    sr = 44100.0
    fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft).tolist()

    print("FFT frequencies results:")
    print("n_fft: ", n_fft)
    print("sr: ", sr)
    print("fft_frequencies: ", fft_frequencies[:8])

def make_mel_bands_weights_files():
    mel_bands_weights_results(40,512,44100,False)
    mel_bands_weights_results(40,512,44100,True)

def mel_bands_weights_results(n_mels: int, n_fft: int, sr: int, htk: bool):
    mel_weights = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=htk,fmin=20.0,fmax=20000.0)
    
    print("Shape of mel weights:", mel_weights.shape)
    mel_weights = mel_weights.tolist()

    print("Mel bands weights results:")
    print("n_mels: ", n_mels)
    print("n_fft: ", n_fft)
    print("sr: ", sr)

    with open(f"examples/tests/results_for_testing_against/librosa_mel_bands_weights_results_nmels={n_mels}_fftsize={n_fft}_sr={sr}_htk={htk}.csv", "w") as f:
        for row in range(len(mel_weights)):
            for col in range(len(mel_weights[row])):
                f.write(f"{mel_weights[row][col]}\n")

# np_linspace_results()
# mel_frequencies_results()
# fft_frequencies_results()
# make_mel_bands_weights_files()