import numpy as np
import matplotlib.pyplot as plt

from scipy import signal, optimize

class SurfaceTexture():
    def __init__(self, raw_data, short_cutoff, long_cutoff):
        self.raw_data = raw_data
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.primary = np.loadtxt(self.raw_data, delimiter=",", unpack=True)
        
        # calc sampling frequency
        self.Fs = 1 / self.primary[0][1] - self.primary[0][0]
        # filter primary using short wave cutoff
        prim_buff, denoised_primary = self.gauss_filt(self.primary[1],
                                                      1 / self.short_cutoff)
        # filter out wavinesss using long wave cutoff                                          
        wav_buff, wavi = self.gauss_filt(denoised_primary,
                                         1 / self.long_cutoff)
        wav_x = self.primary[0][prim_buff:-prim_buff+1][wav_buff:-wav_buff+1]
        
        self.roughness = [wav_x, 
                          denoised_primary[wav_buff:-wav_buff+1] - wavi]
        self.waviness = [wav_x, wavi]
        # generate roughness parameters
        len_wav_x = len(wav_x)
        self.params = {}
        self.params['Ra'] = (sum([abs(x) for x in self.roughness[1]]) 
                            / len_wav_x, "μm")
        self.params['Rq'] = (np.sqrt(sum([x**2 for x in self.roughness[1]])
                            / len_wav_x), "μm")

    def gauss_filt(self, data, cutoff):
        sigma = self.Fs / (2 * np.pi * cutoff)
        window = signal.windows.gaussian(self.Fs * cutoff, sigma) / \
                 (sigma * np.sqrt(2 * np.pi))
        window = [x for x in window if x > 0]
        return len(window)//2, np.convolve(window, data, mode="valid")
    
    def plot_roughness(self, y_lim=None):
        x_lim = (self.primary[0][0], self.primary[0][-1])
        print(f'x_lim = {x_lim}')

        _, axs = plt.subplots(2, 1, figsize=(9, 4))
        
        # plot primary and waviness together
        axs[0].plot(*self.primary, linewidth=0.5, color="blue")
        axs[0].plot(*self.waviness, linewidth=0.5, color="red")
        axs[0].set_title(f"Primary + Waviness, λc = {self.long_cutoff}mm")
        
        # plot roughness
        axs[1].plot(*self.roughness, linewidth=0.5, color="green")
        title = f"Roughness, λc/s = {self.long_cutoff}mm " + \
                f"{self.short_cutoff*1000}μm"
        axs[1].set_title(title)

        for a in axs:
            a.set_xlabel("size, mm")
            a.set_ylabel("height, μm")
            a.set_xlim(x_lim)
            if y_lim: a.set_ylim(y_lim)

        p = ''
        for key in self.params:
            p += f"\n{key} = {self.params[key][0]:.3f}{self.params[key][1]}"

        axs[1].text(0.85, 0.1, p, 
                 transform=axs[1].transAxes, fontsize=8,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', 
                 alpha=0.5))

        plt.tight_layout()
        plt.show()

    def material_ratio(self, samples, Pk_Offset = 1, Vy_Offset = 1):
        peak = max(self.roughness[1]) * (100-Pk_Offset)/100
        valley = min(self.roughness[1]) * (100-Vy_Offset)/100
        Rzmax = peak - valley
        T = Rzmax / samples
        size = len(self.roughness[1])
        #offset_roughness = [x - Rv for x in roughness[1]]
        #size_offset_roughness = len(offset_roughness)

        material_ratio = [], []
        for i in range(samples):
            eval_line = T * i
            material = 0
            for point in self.roughness[1]:
                if point-valley >= eval_line:
                    material += 1

            material_ratio[0].append(eval_line + valley)
            material_ratio[1].append(100 * material / size)
        
        height, ratio = material_ratio
        buff = 1
    
        slope = [], []
        for i in range(buff//2, len(height)- buff//2 - 1):
            slope[0].append(i)
            slope[1].append((height[i+buff] - height[i-buff]) / 
                             ratio[i+buff] - ratio[i-buff])

        plt.subplot(211)
        plt.plot(ratio, height, 'b')
        plt.subplot(212)
        plt.plot(*slope, 'r')
        plt.show()

    def __str__(self):
        p = f"Processed {self.raw_data} with λc = {self.long_cutoff}mm " + \
            f"and λc/s = {self.short_cutoff*1000}μm\n\nParam\tValue"
        for key in self.params:
            p += f"\n{key}:\t{self.params[key][0]:.3f}{self.params[key][1]}"
        p += "\n"
        return p


if __name__ == "__main__":
    data = "example_trace.txt"
    short_cutoff = 2.5 / 1000
    long_cutoff = 0.8
    surface_texture = SurfaceTexture(data, short_cutoff, long_cutoff)
    print(surface_texture)
    #surface_texture.plot_roughness(y_lim=(-1.5, 0.5))
    surface_texture.material_ratio(1000, 5, 5)