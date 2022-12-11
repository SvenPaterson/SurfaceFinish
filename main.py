import matplotlib.pyplot as plt
import numpy as np

from scipy import fft, signal
from scipy.signal import butter, filtfilt


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


def calc_Ra(data):
    return sum([abs(x) for x in data]) / len(data)


def process_roughness(file, short_cutoff, long_cutoff, y_lim = 0):
    long_filt_freq = 1 / long_cutoff
    short_filt_freq = 1 / short_cutoff
    x, primary = np.loadtxt(file, delimiter=",", unpack=True)

    T = x[1] - x[0]
    Fs = 1 / T
    
    prim_buff, denoised_primary = gaussian_filter(primary, Fs, short_filt_freq)
    wav_buff, waviness = gaussian_filter(denoised_primary, Fs, long_filt_freq)
    wav_x = x[prim_buff:-prim_buff+1][wav_buff:-wav_buff+1]

    primary = [x, primary]
    roughness = [wav_x, denoised_primary[wav_buff:-wav_buff+1] - waviness]
    waviness = [wav_x, waviness]
    
    return primary, waviness, roughness


def plot_roughness(data, short_cutoff, long_cutoff, y_lim = 0):
    primary, waviness, roughness = process_roughness(data, short_cutoff, long_cutoff, y_lim)
    x_lim = (primary[0][0], primary[0][-1])

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4))
    
    # plot primary and waviness
    ax1.plot(*primary, linewidth=0.5, color="blue")
    ax1.plot(*waviness, linewidth=0.5, color="red")
    # format 1st plot
    ax1.set_title(f"Primary + Waviness, λc = {long_cutoff}mm")
    ax1.set_xlabel("Length, mm")
    ax1.set_ylabel("Height, μm")
    ax1.set_xlim(x_lim)
    if y_lim: ax1.set_ylim(y_lim)

    # plot roughness
    ax2.plot(*roughness, linewidth=0.5, color="green")
    # format 2nd plot
    ax2.set_title(f"Roughness, λc/s = {long_cutoff}mm / {short_cutoff*1000}μm")
    ax2.set_xlabel("Length, mm")
    ax2.set_ylabel("Height, μm")
    ax2.set_xlim(x_lim)
    if y_lim: ax2.set_ylim(y_lim)

    ax2.text(0.85, 0.1, f"Ra = {calc_Ra(roughness[1]):.3f}μm", transform=ax2.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def material_ratio(roughness, samples):
    Rp = max(roughness[1])
    Rv = min(roughness[1])
    Rzmax = Rp - Rv
    T = Rzmax / samples

    offset_roughness = [x - Rv for x in roughness[1]]
    size_offset_roughness = len(offset_roughness)
    material = 0
    air = 0
    material_ratio = [], []
    for i in range(samples):
        eval_line = T * i
        for point in offset_roughness:
            if point <= eval_line:
                material += 1
            else:
                air += 1
        material_ratio[0].append(eval_line)
        material_ratio[1].append(material / size_offset_roughness)
    
    y, x = material_ratio
    plt.plot(x, y)
    plt.ylim(0, 2)
    plt.show()
        
        


def main():
    long_cutoff = 0.8 #mm
    short_cutoff = 2.5 / 1000 #mm (i.e. microns)
    y_lim = (-1.5, 0.5)

    primary, waviness, roughness = process_roughness("example_trace.txt", short_cutoff, long_cutoff, y_lim)

    material_ratio(roughness, 100)
    
    # current execution time is approx. 0.3s
    plot_roughness("example_trace.txt", short_cutoff, long_cutoff, y_lim)


if __name__ == "__main__":
    main()