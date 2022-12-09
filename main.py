import matplotlib.pyplot as plt
import numpy as np

from scipy import fft, signal
from scipy.signal import butter, filtfilt
from datetime import datetime

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def gaussian_filter(data, Fs, cutoff):
    sigma = Fs / (2 * np.pi * cutoff)
    window = signal.windows.gaussian(Fs * cutoff, sigma) / \
             (sigma * np.sqrt(2 * np.pi))
    window = [x for x in window if x > 0]
    print(f'cutoff, len(window) = {cutoff, len(window)}')
    return len(window)//2, np.convolve(window, data, mode="valid")


def plot_roughness(data, short_cutoff, long_cutoff, y_lim):
    long_filt_freq = 1 / long_cutoff
    short_filt_freq = 1 / short_cutoff
    x, primary = np.loadtxt(data, delimiter=",", unpack=True)
    x_lim = (x[0], x[-1])

    T = x[1] - x[0]
    Fs = 1 / T
    
    prim_buff, denoised_primary = gaussian_filter(primary, Fs, short_filt_freq)
    prim_x = x[prim_buff:-prim_buff+1]
    wav_buff, waviness = gaussian_filter(denoised_primary, Fs, long_filt_freq)
    wav_x = prim_x[wav_buff:-wav_buff+1]
    roughness = denoised_primary[wav_buff:-wav_buff+1] - waviness

    plt.subplot(211)
    # plot primary and waviness
    plt.plot(x, primary, linewidth=0.5, color="blue")
    plt.plot(wav_x, waviness, linewidth=0.5, color="red")
    # format 1st plot
    plt.title(f"Primary + Waviness, λc = {long_cutoff}mm")
    plt.xlabel("Length, mm")
    plt.ylabel("Height, μm")
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.subplot(212)
    # plot roughness
    plt.plot(wav_x, roughness, linewidth=0.5, color="green")
    # format 2nd plot
    plt.title(f"Roughness, λc/s = {long_cutoff}mm / {short_cutoff*1000}μm")
    plt.xlabel("Length, mm")
    plt.ylabel("Height, μm")
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.tight_layout()
    #plt.savefig()


def main():
    long_cutoff = 0.8 #mm
    short_cutoff = 2.5 / 1000 #mm (i.e. microns)
    y_lim = (-1.5, 0.5)
    
    start = datetime.now()
    print(f'start: {start}')
    plot_roughness("example_trace.txt", short_cutoff, long_cutoff, y_lim)
    print(f'end: {datetime.now() - start}')


    #y_filtl = butter_lowpass_filter(y, long_filt_freq, Fs) 
    #y_filth = butter_highpass_filter(y, long_filt_freq, Fs)

    #calc FFT of raw data
    #y_fft = fft.fft(y)
    #y_freq = fft.fftfreq(N, T)[:N//2]

    # plt.plot(y_freq, 2.0/N * np.abs(y_fft[:N//2]), linewidth=0.5)
    #plt.plot(x, y - y_filtl, linewidth=0.5, color="blue")
    #plt.plot(x, y_filth, linewidth=0.5, color="purple")

if __name__ == "__main__":
    main()