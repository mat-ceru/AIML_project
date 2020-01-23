import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from google.colab import files
import imageio
import numpy as np
import librosa
import librosa.display
#uploaded = files.upload()
'''
cmap = plt.get_cmap('jet')

sample_rate, samples = wavfile.read('/content/03-02-01-01-01-01-01.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
imageio.imwrite("spectogram_bw.png", (65535*(1-spectrogram)).astype(np.uint16))
imageio.imwrite("spectogram_color.png", (255*(cmap(spectrogram))).astype(np.uint8)[:,:,:3])

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel("Freq [Hz]")
plt.xlabel('Time [sec]')
plt.show()
'''

y, sr = librosa.load("/content/aaron_neville-tell_it_like_it_is.mp3")
sr2 = sr

array = []

f = open("/content/aaron_neville-tell_it_like_it_is.mel")
lines = f.readlines()

idx = 0
for line in lines:
  values = line.split(",")
  for value in values:
    #idx = idx + 1
    #if(idx <= 2000):
    array.append(float(value))

narray = np.array(array)

#librosa.feature.melspectrogram(y=narray, sr=sr)
# array([[  2.891e-07,   2.548e-03, ...,   8.116e-09,   5.633e-09],
# [  1.986e-07,   1.162e-02, ...,   9.332e-08,   6.716e-09],
# ...,
# [  3.668e-09,   2.029e-08, ...,   3.208e-09,   2.864e-09],
# [  2.561e-10,   2.096e-09, ...,   7.543e-10,   6.101e-10]])

# Using a pre-computed power spectrogram would give the same result:

D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D, sr=sr)

# Display of mel-frequency spectrogram coefficients, with custom
# arguments for mel filterbank construction (default is fmax=sr/2):

# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)
plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000)
#plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=16000)
#plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

D = np.abs(librosa.stft(narray))**2
S = librosa.feature.melspectrogram(S=D, sr=sr2)

# Display of mel-frequency spectrogram coefficients, with custom
# arguments for mel filterbank construction (default is fmax=sr/2):

# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=narray, sr=sr2, n_mels=128,
                                    fmax=8000)
plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr2,
                         fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr2,
                         fmax=16000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()
