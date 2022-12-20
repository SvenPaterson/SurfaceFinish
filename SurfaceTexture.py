import time, os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from scipy import signal, optimize
from functools import wraps
from blume.table import table

class SurfaceTexture():
    
    def timeit(method):
        @wraps(method)
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print(f"{method.__name__} took {te-ts:.3f} seconds")
            return result
        return timed

    def __init__(self, raw_data: str, short_cutoff: int, 
                 long_cutoff: float, order=1, units='mm', **kwargs):
        """ Process surface texture data from Taylor Hobson Talysurf.
            Currently roughness parameters are calculated on init. Methods can be
            called to plot roughness and material ratio properties.

        Args:
            raw_data (str):        Input is .txt file from Taylor Hobson Talysurf, 
                                   1st column is x, 2nd column is y
            short_cutoff (int): short wave cutoff in micron, default is 8 micron
            long_cutoff (float):  long wave cutoff in mm, default is 0.8mm
            order (int, optional): Determines order of least mean squares regression 
                                   to remove initial form for trace if, for example, 
                                   the measured surface is sloped, curved etc.
                                   Defaults to 1 (i.e. fit trace to line of y = mx + b).
                                   Set to 0 to skip leveling.
            **kwargs:              
                PLOT_LEVEL
                PLOT_MR
                PLOT_ROUGHNESS
                PLOT_ALL
        """

        kwargs.setdefault('PLOT_LEVEL', False)
        kwargs.setdefault('PLOT_MR', False)
        kwargs.setdefault('PLOT_ROUGHNESS', False)
        kwargs.setdefault('PLOT_ALL', False)

        self.units = units
        self.raw_data = raw_data
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.primary = np.loadtxt(self.raw_data, delimiter=",", unpack=True)

        # if profile leveling is called for then fit to line/curve
        if order: 
            if order > 3:
                raise ValueError("order must be 1, 2 or 3 (0 to skip leveling)")
            self.order = order
            def first_order_func(x, a, b):
                return a * x + b
            def second_order_func(x, a, b, c):
                return a * x ** 2 + b * x + c
            def third_order_func(x, a, b, c, d):
                return a * x ** 3 + b * x ** 2 + c * x + d
            case = {1: first_order_func,
                    2: second_order_func,
                    3: third_order_func}
            func = case[self.order]

            x = self.primary[0]
            y = self.primary[1]
            popt, _ = optimize.curve_fit(func, x, y)
            if kwargs['PLOT_LEVEL'] or kwargs['PLOT_ALL']:
                if self.order == 1:
                    plt.plot(x, func(x, *popt), 'r-',
                            label='fit: a=%5.10f, b=%5.10f' % tuple(popt))
                else:
                    plt.plot(x, func(x, *popt), 'r-')
                plt.title("Initial Profile Leveling")
                plt.plot(x, y, 'b-', label='data')
                plt.legend()
                plt.show()
            self.primary = np.vstack((np.array(x),
                                      np.array(y - func(x, *popt))))


        # calc sampling frequency
        self.Fs = 1 / self.primary[0][1] - self.primary[0][0]

        def gauss_filt(data, cutoff):
            sigma = self.Fs / (2 * np.pi * cutoff)
            window = signal.windows.gaussian(self.Fs * cutoff, sigma) /\
                    (sigma * np.sqrt(2 * np.pi))
            window = [x for x in window if x > 0]
            return len(window)//2, np.convolve(window, data, mode="valid")

        # filter primary using short wave cutoff
        prim_buff, denoised_primary = gauss_filt(self.primary[1],
                                                 1 / self.short_cutoff)
        # filter out wavinesss using long wave cutoff                                          
        wav_buff, waviness = gauss_filt(denoised_primary, 1 / self.long_cutoff)
        wav_x = self.primary[0][prim_buff:-prim_buff+1][wav_buff:-wav_buff+1]
        self.roughness = np.vstack((wav_x,
                                    denoised_primary[wav_buff:-wav_buff+1]
                                    - waviness))
        self.waviness = np.vstack((wav_x, waviness))

        # generate roughness parameters
        len_wav_x = len(wav_x)
        self.R_params = {}
        self.R_params['Ra'] = (sum(map(abs, self.roughness[1]))
                            / len_wav_x, "μm")
        self.R_params['Rq'] = (np.sqrt(sum(map(np.square, self.roughness[1]))
                            / len_wav_x), "μm")
        self.R_params['Rsk'] = (sum([x ** 3 for x in self.roughness[1]]) /
                             ((self.R_params['Rq'][0] ** 3) * len_wav_x), "")
        self.R_params['Rku'] = (sum([x ** 4 for x in self.roughness[1]]) /
                             ((self.R_params['Rq'][0] ** 4) * len_wav_x), "")
        self.R_params['Rt'] = (max(self.roughness[1]) - min(self.roughness[1]), 
                            "μm")
        
        if kwargs['PLOT_ROUGHNESS'] or kwargs['PLOT_ALL']:
            self.plot_roughness()

        if kwargs['PLOT_MR'] or kwargs['PLOT_ALL']:
            self.plot_material_ratio()

    def plot_roughness(self, y_lim=None):
        fig, axs = plt.subplots(2, 1, figsize=(9, 4))
        
        # plot primary and waviness together
        axs[0].plot(*self.primary, linewidth=0.5, color="blue")
        axs[0].plot(*self.waviness, linewidth=0.5, color="red")
        PW_title = f"Primary + Waviness, λc/s = {self.long_cutoff}mm " +\
                   f"{self.short_cutoff*1000}μm"
        axs[0].set_title(PW_title)
        
        # plot roughness
        axs[1].plot(*self.roughness, linewidth=0.5, color="green")
        R_title = f"Roughness, λc/s = {self.long_cutoff}mm " +\
                f"{self.short_cutoff*1000}μm"
        axs[1].set_title(R_title)

        minor_locator = AutoMinorLocator(2)
        # set x and y limits
        x_lim = (self.primary[0][0], self.primary[0][-1])
        for a in axs:
            a.set_xlabel("size, mm")
            a.set_ylabel("height, μm")
            a.set_xlim(x_lim)
            if y_lim: a.set_ylim(y_lim)
            a.axhline(0, color="black", linewidth=0.5)
            a.grid(True, which='both',
                   axis='both',
                   linewidth=0.5,
                   linestyle=(0, (5, 10)),
                   color='grey', 
                   alpha=0.5)
            #a.minorticks_on()
            a.yaxis.set_minor_locator(minor_locator)
            a.xaxis.set_minor_locator(minor_locator)

        # add R-Parameters to plot
        p = 'ISO 21290-2:2021\n'
        for key in self.R_params:
            p += f"{key} = {self.R_params[key][0]:.3f}{self.R_params[key][1]}\n"
        fig.text(0.02, 0.05, p, 
                    transform=axs[1].transAxes, fontsize=8,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', 
                    alpha=0.5))

        plt.tight_layout()
        plt.show()

    @timeit
    def get_material_ratio(self, samples=2000, Pk_Offset=0.01, Vy_Offset=0.01):
        self.mr_params = {}
        self.Pk_Offset, self.Vy_Offset = Pk_Offset, Vy_Offset

        # sort the uniformly sampled profile in descending order
        self.material_ratio = np.sort(self.roughness[1])[::-1]
        self.material_ratio_all = self.roughness[:, self.roughness[1].argsort()[::-1]]
        
        # the sampling distance
        deltaX = self.material_ratio.size / samples
        interpolated_material_ratio = np.zeros((2, samples))
        for i in range(samples):
            # find the index of the profile that is closest to the 
            # interpolated value
            index = int(i * deltaX)
            interpolated_material_ratio[0][i] = 100 * i / samples
            interpolated_material_ratio[1][i] = self.material_ratio[index]
        self.material_ratio = np.asarray(interpolated_material_ratio)

        # calc best fit straight line which includes 40% of measured points
        delta40 = int(0.4 * samples)
        bf40_grad = float('inf') # best fit gradient of 40% kernel
        for i in range(samples - delta40):
            x = self.material_ratio[0][i:i + delta40]
            y = self.material_ratio[1][i:i + delta40]
            # use least square line instead of secant
            m, c = np.polyfit(x, y, 1)
            if abs(m) < bf40_grad:
                bf40_grad = abs(m)
                self.bf40_eq = (m, c)

        # y = mx + c
        self.b40at100 = self.bf40_eq[0] * 100 + self.bf40_eq[1]
        self.mr_params['Rvkx'] = self.b40at100 - self.material_ratio[1][-1] 
        self.mr_params['Rk'] = self.bf40_eq[1] - self.b40at100
        # intersection of self.roughness and best fit line
        self.mr_params['Rpkx'] = self.material_ratio[1][0] - self.bf40_eq[1]

        self.mr_params['Rak1'] = 0
        self.mr_params['Rak2'] = 0
                ## DOUBLE CHECK THIS CALCULATION ##
        for i, h in enumerate(self.material_ratio[1]):
            if h < self.bf40_eq[1]:
                self.mr_params['Rmrk1'] = self.material_ratio[0][i]
                self.Rmrk1_y = self.material_ratio[1][i]
                self.mr_params['Rak1'] += (self.material_ratio[1][i]
                                           - self.bf40_eq[1])
                break
        self.Rak1_points = [], []
        self.Rak2_points = [], []
        for i, h in enumerate(self.material_ratio[1][::-1]):
            if h > self.b40at100:
                self.mr_params['Rmrk2'] = self.material_ratio[0][::-1][i]
                self.Rmrk2_y = self.material_ratio[1][::-1][i]
                
                break

        # calculate Rak values
        count = 0
        for i, h in enumerate(self.material_ratio_all[1]):
            x = self.material_ratio_all[0][i]
            y = self.material_ratio_all[1][i]
            if h > self.bf40_eq[1]:
                self.Rak1_points[0].append(x)
                self.Rak1_points[1].append(h)
                self.mr_params['Rak1'] += (y - self.bf40_eq[1])
                count += 1
            else:
                self.Rak1_points[0].append(x)
                self.Rak1_points[1].append(np.nan)
        self.mr_params["Rak1"] = self.mr_params["Rak1"] / \
                                 ((count)*100/len(self.roughness[0]))
        count = 0
        for i, h in enumerate(self.material_ratio_all[1][::-1]):
            x = self.material_ratio_all[0][::-1][i]
            y = self.material_ratio_all[1][::-1][i]
            if h < self.b40at100:
                self.Rak2_points[0].append(x)
                self.Rak2_points[1].append(h)
                self.mr_params['Rak2'] += (y - self.b40at100)
                count += 1
            else:
                self.Rak2_points[0].append(x)
                self.Rak2_points[1].append(np.nan)
        self.mr_params["Rak2"] = self.mr_params["Rak2"] /\
                                 ((count)*100/len(self.roughness[0]))
        
        print(f'Rak1: {self.mr_params["Rak1"]}')
        print(f'Rak2: {self.mr_params["Rak2"]}')
        
        self.Rak1_points = np.asarray(self.Rak1_points)
        self.Rak2_points = np.asarray(self.Rak2_points)
        self.Rak1_points = self.Rak1_points[:, self.Rak1_points[0, :].argsort()]
        self.Rak2_points = self.Rak2_points[:, self.Rak2_points[0, :].argsort()]

        self.mr_params['Rak2'] = abs(self.mr_params['Rak2'])

        self.mr_params['Rpk'] = 2 * self.mr_params['Rak1'] / \
                                self.mr_params['Rmrk1']

        self.mr_params['Rvk'] = 2 * self.mr_params['Rak2'] / \
                                (100 - self.mr_params['Rmrk2'])


    def plot_material_ratio(self):
        self.get_material_ratio()
        fig, axs = plt.subplots(nrows=2, ncols=2,
                                figsize=(12, 6),
                                gridspec_kw={'width_ratios': [2, 1],
                                             'height_ratios': [30, 1]})

        # combine bottom two axes and create space for R-Parameters
        gs = axs[1, 1].get_gridspec()
        for a in axs[1,:]:
            a.set_axis_off()
        #axbig = fig.add_subplot(gs[1, :])
        #axbig.set_axis_off()
        
        # plot roughness
        axs[0, 0].plot(*self.roughness, linewidth=0.5, color="blue")
        axs[0, 0].set_xlim(self.roughness[0][0], self.roughness[0][-1])
        axs[0, 0].plot(*self.Rak1_points, linewidth=0.5, color="red")
        axs[0, 0].plot(*self.Rak2_points, linewidth=0.5, color="red")

        # plot Rpk and Rvk lines
        Rpk_line = self.material_ratio[1][0] - self.mr_params['Rpk']
        Rvk_line = self.material_ratio[1][-1] + self.mr_params['Rvk']
        axs[0, 0].axhline(y=Rpk_line, linestyle='--', 
                          color="green", linewidth=0.5)
        axs[0, 0].axhline(y=Rvk_line, linestyle='--',
                          color="green", linewidth=0.5)

        # plot material ratio
        axs[0, 1].plot(*self.material_ratio, color="red")
        for xlabel_i in axs[0, 1].get_yticklabels():
            xlabel_i.set_visible(False)
        x = np.linspace(0, 100, 100)
        y = self.bf40_eq[0] * x + self.bf40_eq[1]
        axs[0, 1].plot(x, y, color="green", linewidth=0.5)
        axs[0, 1].set_xlim(0, 100)
        #plot a dot at Rmrk1 & Rmrk2
        axs[0, 1].plot(self.mr_params['Rmrk1'], self.Rmrk1_y,
                       'x', color="green")
        axs[0, 1].plot(self.mr_params['Rmrk2'], self.Rmrk2_y,
                       'x', color="green")

        for a in axs[0, :]:
            a.minorticks_on()
            a.set_ylim(round(min(self.material_ratio[1]) - .2, 2), 
                       round(max(self.material_ratio[1]) + .2, 2))
            a.axhline(0, color="black", linewidth=0.5)
            # plot 40% kernel lines
            a.axhline(self.bf40_eq[1], linestyle='--',
                      color="green", linewidth=0.5)
            a.axhline(self.b40at100, linestyle='--',
                      color="green", linewidth=0.5)

        # create Rmr parameters table
        rows = ['Rpkx', 'Rk', 'Rvkx', 'Rmrk1', 'Rmrk2', 'Rak1', 'Rak2']
        units = ['μm', 'μm', 'μm', '%', '%', 'μm * %', 'μm * %']
        columns = ['ISO 21290-2:2021 Parameter', 'Value', 'Unit']
        n_rows = len(rows)
        
        cell_text = []
        for i in range(n_rows):
            cell_text.append([round(self.mr_params[rows[i]], 3), units[i]])
        the_table = table(axs[1, 1], cellText=cell_text,
                                 rowLabels=rows,
                                 colLabels=columns,
                                 loc='top',
                                 fontsize=10)
        
        axs[0, 0].set_title(f"{os.path.split(self.raw_data)[1]} - Roughness")
        axs[0, 1].set_title("Material Ratio")
        plt.tight_layout()
        plt.show()

    @timeit
    def old_material_ratio(self, samples, Pk_Offset=0.01, Vy_Offset=0.01,
                           plot_flag=False):
        PLT_SLOPE = True # do you want to plot the slope of the material ratio?

        # should have reviewed ISO 21920-2:2021 before writing this!
        # see Secton 4.5.1.3 and Annex C for appropriate calculation
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
                (material_ratio[0][i] - material_ratio[0][i + dx]) /\
                (material_ratio[1][i] - material_ratio[1][i + dx])

        # Find location of minimum slope and mr_params['Rk'] params at that point
        index_max = max(range(100, len(mr_slope_y)-100),
                        key=mr_slope_y.__getitem__)
        Rk_slope = mr_slope_y[index_max]
        Rk_loc = np.where(material_ratio[1] == mr_slope_x[index_max])
        Rk_point = (mr_slope_x[index_max], material_ratio[0][Rk_loc])

        # y = mx + b, b = y - mx
        b = Rk_point[1] - Rk_slope * Rk_point[0]
        Rk_line = (np.linspace(0, 100, 100), 
                  Rk_slope * np.linspace(0, 100, 100) + b)

        if plot_flag:
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
        p = f"Processed {self.raw_data} with λc = {self.long_cutoff}mm " +\
            f"and λc/s = {self.short_cutoff*1000}μm\n\nParam\tValue"
        for key in self.R_params:
            p += f"\n{key}:\t{self.R_params[key][0]:.3f}{self.R_params[key][1]}"
        p += "\n"
        return p


if __name__ == "__main__":
    data = "chrome_example\chrome_rod.txt"
    short_cutoff = 2.5 / 1000
    long_cutoff = 0.8
    surface_texture = SurfaceTexture(data, short_cutoff, long_cutoff, order=2, 
                                     PLOT_MR=True)
    #print(surface_texture)
    # time the functions
    #surface_texture.old_material_ratio(1000)
