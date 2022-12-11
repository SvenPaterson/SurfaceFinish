import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

class Roughness():
    def __init__(self, raw_data, short_cutoff, long_cutoff):
        self.raw_data = raw_data
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.primary = np.loadtxt(self.raw_data, delimiter=",", unpack=True)
        self.process_roughness()


    def gauss_filt(self, data, cutoff):
        sigma = self.Fs / (2 * np.pi * cutoff)
        window = signal.windows.gaussian(self.Fs * cutoff, sigma) / \
                 (sigma * np.sqrt(2 * np.pi))
        window = [x for x in window if x > 0]
        return len(window)//2, np.convolve(window, data, mode="valid")
    

    def calc_Ra(self, rough):
        return 
    

    def process_roughness(self):
        self.Fs = 1 / self.primary[0][1] - self.primary[0][0]
        prim_buff, denoised_primary = self.gauss_filt(self.primary[1],
                                                      1 / self.short_cutoff)
        wav_buff, wavi = self.gauss_filt(denoised_primary,
                                             1 / self.long_cutoff)
        wav_x = self.primary[0][prim_buff:-prim_buff+1][wav_buff:-wav_buff+1]
    
        self.rough = [wav_x, denoised_primary[wav_buff:-wav_buff+1] - wavi]
        self.wavi = [wav_x, wavi]
        self.Ra = sum([abs(x) for x in self.rough[1]]) / len(self.rough[1])


    
    def plot_roughness(self, y_lim=None):
        x_lim = (self.primary[0][0], self.primary[0][-1])

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4))
        
        # plot primary and wavi
        ax1.plot(*self.primary, linewidth=0.5, color="blue")
        ax1.plot(*self.wavi, linewidth=0.5, color="red")
        # format 1st plot
        ax1.set_title(f"Primary + Waviness, λc = {self.long_cutoff}mm")
        ax1.set_xlabel("size, mm")
        ax1.set_ylabel("Height, μm")
        ax1.set_xlim(x_lim)
        if y_lim: ax1.set_ylim(y_lim)

        # plot rough
        ax2.plot(*self.rough, linewidth=0.5, color="green")
        # format 2nd plot
        title = f"Roughness, λc/s = {self.long_cutoff}mm " + \
                f"{self.short_cutoff*1000}μm"
        ax2.set_title(title)
        ax2.set_xlabel("size, mm")
        ax2.set_ylabel("Height, μm")
        ax2.set_xlim(x_lim)
        if y_lim: ax2.set_ylim(y_lim)

        ax2.text(0.85, 0.1, f"Ra = {self.Ra:.3f}μm", 
                 transform=ax2.transAxes, fontsize=8,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', 
                 alpha=0.5))

        plt.tight_layout()
        plt.show()

    def __str__(self):
        p = f"Processed {self.raw_data} with λc = {self.long_cutoff}mm " + \
            f"and λc/s = {self.short_cutoff*1000}μm\n\t Ra: {self.Ra:.3f}μm"
        return p

if __name__ == "__main__":
    data = "example_trace.txt"
    short_cutoff = 2.5 / 1000
    long_cutoff = 0.8
    surf_finish = Roughness(data, short_cutoff, long_cutoff)
    print(surf_finish)
    surf_finish.plot_roughness()