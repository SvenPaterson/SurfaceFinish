import numpy as np
import matplotlib.pyplot as plt

from scipy import signal, optimize

class SurfaceTexture():
    def __init__(self, raw_data: str, short_cutoff: int, 
                 long_cutoff: float, order=1):
        """ Process surface texture data from Taylor Hobson Talysurf.
            Currently roughness parameters are calculated on init. Methods can be
            called to plot roughness and material ratio properties.

        Args:
            raw_data (str):        Input is .txt file from Taylor Hobson Talysurf, 
                                   1st column is x, 2nd column is y
            short_cutoff (_type_): short wave cutoff in micron, default is 8 micron
            long_cutoff (_type_):  long wave cutoff in mm, default is 0.8mm
            order (int, optional): Determines order of least mean squares regression 
                                   to remove initial form for trace if, for example, 
                                   the measured surface is sloped, curved etc.
                                   Defaults to 1 (i.e. fit trace to line of y = mx + b).
        """
        self.raw_data = raw_data
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.primary = np.loadtxt(self.raw_data, delimiter=",", unpack=True)
        if order: #for LMS regression, flatten trace first
            self.order = order
            self.primary = self.LMS_Regr(self.primary)
        
        # calc sampling frequency
        self.Fs = 1 / self.primary[0][1] - self.primary[0][0]
        # filter primary using short wave cutoff
        prim_buff, denoised_primary = self.gauss_filt(self.primary[1],
                                                      1 / self.short_cutoff)
        # filter out wavinesss using long wave cutoff                                          
        wav_buff, waviness = self.gauss_filt(denoised_primary,
                                         1 / self.long_cutoff)
        wav_x = self.primary[0][prim_buff:-prim_buff+1][wav_buff:-wav_buff+1]
  
        self.roughness = [wav_x, denoised_primary[wav_buff:-wav_buff+1]
                          - waviness]
        self.waviness = [wav_x, waviness]
        # generate roughness parameters
        len_wav_x = len(wav_x)

        self.params = {}
        self.params['Ra'] = (sum(map(abs, self.roughness[1])) / len_wav_x, "μm")
        self.params['Rq'] = (np.sqrt(sum(map(np.square, self.roughness[1]))
                            / len_wav_x), "μm")
        self.params['Rsk'] = (sum([x ** 3 for x in self.roughness[1]]) / \
                             ((self.params['Rq'][0] ** 3) * len_wav_x), "")
        self.params['Rku'] = (sum([x ** 4 for x in self.roughness[1]]) / \
                             ((self.params['Rq'][0] ** 4) * len_wav_x), "")
        self.params['Rt'] = (max(self.roughness[1]) - min(self.roughness[1]), "μm")

    def LMS_Regr(self, func):

        if self.order < 1 or self.order > 3:
            raise ValueError("order must be 1, 2 or 3")
        def first_order_func(x, a, b):
            return a * x + b
        def second_order_func(x, a, b, c):
            return a * x ** 2 + b * x + c
        def third_order_func(x, a, b, c, d):
            return a * x ** 3 + b * x ** 2 + c * x + d
        case = {1: first_order_func, 2: second_order_func, 3: third_order_func}
        func = case[self.order]
        
        x = self.primary[0]
        y = self.primary[1]
        popt, pcov = optimize.curve_fit(func, x, y)
        # print(popt)
        # print(pcov)
        plt.plot(x, func(x, *popt), 'r-')
        #          label='fit: a=%5.10f, b=%5.10f' % tuple(popt))
        plt.plot(x, y, 'b-', label='data')
        plt.legend()
        plt.show()
        return x, y - func(x, *popt)

    def gauss_filt(self, data, cutoff):
        sigma = self.Fs / (2 * np.pi * cutoff)
        window = signal.windows.gaussian(self.Fs * cutoff, sigma) / \
                 (sigma * np.sqrt(2 * np.pi))
        window = [x for x in window if x > 0]
        return len(window)//2, np.convolve(window, data, mode="valid")
    
    def plot_roughness(self, y_lim=None):
        fig, axs = plt.subplots(2, 1, figsize=(9, 4))
        
        # plot primary and waviness together
        axs[0].plot(*self.primary, linewidth=0.5, color="blue")
        axs[0].plot(*self.waviness, linewidth=0.5, color="red")
        axs[0].set_title(f"Primary + Waviness, λc = {self.long_cutoff}mm")
        
        # plot roughness
        axs[1].plot(*self.roughness, linewidth=0.5, color="green")
        title = f"Roughness, λc/s = {self.long_cutoff}mm " + \
                f"{self.short_cutoff*1000}μm"
        axs[1].set_title(title)

        # set x and y limits
        x_lim = (self.primary[0][0], self.primary[0][-1])
        for a in axs:
            a.set_xlabel("size, mm")
            a.set_ylabel("height, μm")
            a.set_xlim(x_lim)
            if y_lim: a.set_ylim(y_lim)

        # add R-Parameters to plot
        p = 'ISO 21290-2:2021\n'
        for key in self.params:
            p += f"{key} = {self.params[key][0]:.3f}{self.params[key][1]}\n"
        fig.text(0.05, 0.05, p, 
                    transform=axs[1].transAxes, fontsize=8,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', 
                    alpha=0.5))

        plt.tight_layout()
        plt.show()

    def material_ratio(self, samples, Pk_Offset = 0.01, Vy_Offset = 0.01):
        PLT_SLOPE = True # do you want to plot the slope of the material ratio?
        # should have reviewed ISO 21920-2:2021 before writing this!
        if 0 < Pk_Offset < .25 and 0 < Vy_Offset < .25:
            pass
        else:
            raise ValueError("Pk_Offset and Vy_Offset must be between 0 and 0.25")

        # calculate material ratio
        peak = max(self.roughness[1]) * (1 - Pk_Offset)
        valley = min(self.roughness[1]) * (1 - Vy_Offset)
        material_ratio = np.zeros((2, samples))
        for i in range(samples):
            eval_line = peak - valley / samples * i
            material = 0
            for point in self.roughness[1]:
                if point-valley >= eval_line:
                    material += 1
            material_ratio[0][i] = eval_line + valley
            material_ratio[1][i] = 100 * material / len(self.roughness[1])

        # calculate slope of material ratio
        dx = 50 # for finding df/dx
        mr_slope_x, mr_slope_y = np.zeros(samples - dx), np.zeros(samples - dx)
        for i in range(samples - dx):
            mr_slope_x[i] = material_ratio[1][i + dx//2]
            mr_slope_y[i] = \
                (material_ratio[0][i] - material_ratio[0][i + dx]) / \
                (material_ratio[1][i] - material_ratio[1][i + dx])

        # Find location of minimum slope and Rk params at that point
        index_max = max(range(100, len(mr_slope_y)-100), key=mr_slope_y.__getitem__)
        Rk_slope = mr_slope_y[index_max]
        Rk_loc = np.where(material_ratio[1] == mr_slope_x[index_max])
        Rk_point = (mr_slope_x[index_max], material_ratio[0][Rk_loc])

        # y = mx + b, b = y - mx
        b = Rk_point[1] - Rk_slope * Rk_point[0]
        Rk_line = (np.linspace(0, 100, 100), 
                  Rk_slope * np.linspace(0, 100, 100) + b)

        # plot material ratio and slope
        if PLT_SLOPE: plt.subplot(211)
        x_lim = (0, 100)
        plt.xlim(x_lim)
        plt.plot(*np.flip(material_ratio), 'b')
        plt.plot(*Rk_line, 'r--')
        if PLT_SLOPE: 
            plt.subplot(212)
            plt.xlim(x_lim)
            plt.ylim(-.01, 0)
            plt.plot(mr_slope_x, mr_slope_y, 'r')
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
    surface_texture = SurfaceTexture(data, short_cutoff, long_cutoff, order=3)
    print(surface_texture)
    surface_texture.material_ratio(1000, Pk_Offset=0.01, Vy_Offset=0.01)