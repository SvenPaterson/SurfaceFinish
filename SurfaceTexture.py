import time, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from scipy import signal, optimize
from functools import wraps
from blume.table import table

class SurfaceTexture():

    @staticmethod
    def _parse_column_selector(selector):
        if isinstance(selector, int):
            return selector
        if isinstance(selector, str) and selector.strip().isdigit():
            return int(selector.strip())
        return selector

    @staticmethod
    def _read_text_profile(raw_data):
        try:
            return np.loadtxt(raw_data, delimiter=",", unpack=True)
        except ValueError:
            return np.loadtxt(raw_data, unpack=True)

    @classmethod
    def _read_excel_profile(cls, raw_data, x_col, y_col, sheet_name=0):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading .xlsx files requires pandas and openpyxl. "
                "Install with: pip install pandas openpyxl"
            ) from exc

        x_selector = cls._parse_column_selector(x_col)
        y_selector = cls._parse_column_selector(y_col)

        df = pd.read_excel(raw_data, sheet_name=sheet_name)
        if isinstance(df, dict):
            first_sheet = next(iter(df))
            df = df[first_sheet]

        if isinstance(x_selector, str):
            if x_selector not in df.columns:
                raise ValueError(f"X column '{x_selector}' was not found in sheet '{sheet_name}'.")
            x_series = df[x_selector]
        else:
            x_series = df.iloc[:, x_selector]

        if isinstance(y_selector, str):
            if y_selector not in df.columns:
                raise ValueError(f"Y column '{y_selector}' was not found in sheet '{sheet_name}'.")
            y_series = df[y_selector]
        else:
            y_series = df.iloc[:, y_selector]

        x = x_series.to_numpy(dtype=float)
        y = y_series.to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            raise ValueError("No valid numeric X/Y pairs found in the selected worksheet columns.")

        return np.vstack((x[valid], y[valid]))

    @classmethod
    def _load_profile_data(cls, raw_data, x_col=0, y_col=1, sheet_name=0):
        ext = Path(raw_data).suffix.lower()
        if ext == ".xlsx":
            return cls._read_excel_profile(raw_data, x_col=x_col, y_col=y_col, sheet_name=sheet_name)
        return cls._read_text_profile(raw_data)
    
    def timeit(method):
        @wraps(method)
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print(f"{method.__name__} took {te-ts:.3f} seconds")
            return result
        return timed

    @timeit
    def __init__(self, raw_data: str, short_cutoff: int,
                 long_cutoff: float, order=1,
                 x_units='mm', y_units='μm',
                 x_col=0, y_col=1, sheet_name=0, **kwargs):
        """ Process surface texture data
            Currently roughness parameters are calculated on init. Methods can be
            called to plot roughness and material ratio properties.

        Args:
            raw_data (str):
                Input is csv file with 2 columns, 1st column is
                x, 2nd column is y
            short_cutoff (int):
                short wave cutoff in micron, default is 8 micron
            long_cutoff (float):
                long wave cutoff in mm, default is 0.8mm
            order (int, optional):
                Determines order of least mean squares regression
                to remove initial form for trace if, for example,
                the measured surface is sloped, curved etc.
                Defaults to 1 (i.e. fit trace to line of y = mx + b).
                Set to 0 to skip leveling.
            **kwargs:              
                PLOT_LEVEL, PLOT_MR, PLOT_ROUGHNESS, PLOT_ALL
                Default is False for all. Set to True to plot.
        """

        kwargs.setdefault('PLOT_LEVEL', False)
        kwargs.setdefault('PLOT_MR', False)
        kwargs.setdefault('PLOT_ROUGHNESS', False)
        kwargs.setdefault('PLOT_ALL', False)

        self.x_units = x_units
        self.y_units = y_units
        self.raw_data = raw_data
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.order = order
        self.primary = self._load_profile_data(
            self.raw_data,
            x_col=x_col,
            y_col=y_col,
            sheet_name=sheet_name,
        )

        # Keep an immutable copy of the raw input for plotting/inspection.
        self.raw_data_xy = self.primary.copy()
        self.level_fit_x = None
        self.level_fit_y = None

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

            # Save the leveling fit so it can be plotted alongside raw data later.
            self.level_fit_x = np.array(x)
            self.level_fit_y = func(x, *popt)

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


        # calc sampling frequency (robust to descending/non-uniform x spacing)
        diffs = np.diff(self.primary[0])
        diffs = np.abs(diffs[np.isfinite(diffs)])
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            raise ValueError("Invalid x spacing: unable to compute positive sample spacing from X data.")
        T = float(np.median(diffs))
        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"Invalid x spacing: computed sample spacing T={T}.")
        self.Fs = 1.0 / T

        def gauss_filter(data, cutoff):
            sigma = abs(self.Fs / (2 * np.pi * cutoff))
            sigma = max(sigma, np.finfo(float).eps)

            window_len = int(round(abs(self.Fs * cutoff)))
            window_len = max(window_len, 3)
            if window_len % 2 == 0:
                window_len += 1

            window = signal.windows.gaussian(window_len, sigma) /\
                    (sigma * np.sqrt(2 * np.pi))
            # plt.plot(window)
            window = np.asarray([x for x in window if x > 0], dtype=float)
            # plt.plot(window, 'r', linestyle='--')
            # plt.show()
            return signal.fftconvolve(data, window, mode="same")

        # buffer used for clipping valid data
        prim_buff = int(self.short_cutoff / (2 * T))
        wav_buff = int(self.long_cutoff / (2 * T))

        # Validate cutoffs vs available data length
        n_pts = self.primary[0].size
        total_buff = 2 * (prim_buff + wav_buff)
        if prim_buff < 1 or wav_buff < 1 or total_buff >= n_pts - 4:
            raise ValueError(
                "Filter cutoffs are incompatible with the data length / spacing.\n"
                f"  Sample spacing T = {T:.6g} {self.x_units}\n"
                f"  Short cutoff = {self.short_cutoff} {self.x_units} -> buffer {prim_buff} samples\n"
                f"  Long cutoff  = {self.long_cutoff} {self.x_units} -> buffer {wav_buff} samples\n"
                f"  Trace length = {n_pts} samples; need < {n_pts - 4} buffer but require {total_buff}.\n"
                "  Tip: cutoffs must be in the same distance units as the X data "
                f"({self.x_units}). For inch data try short=9.8425e-5, long=0.031496."
            )

        # filter primary using short wave cutoff and clip valid data
        freq = 1 / self.short_cutoff
        denoised_primary = gauss_filter(self.primary[1], freq)
        denoised_primary = denoised_primary[prim_buff:-prim_buff+1]

        # filter out wavinesss using long wave cutoff and clip valid data
        freq = 1 / self.long_cutoff                                        
        waviness = gauss_filter(denoised_primary, freq)
        denoised_primary = denoised_primary[wav_buff:-wav_buff+1]
        waviness = waviness[wav_buff:-wav_buff+1]

        # store data in class variables
        wav_x = self.primary[0][prim_buff:-prim_buff+1][wav_buff:-wav_buff+1]
        self.roughness = np.vstack((wav_x, denoised_primary - waviness))
        self.waviness = np.vstack((wav_x, waviness))

        # generate roughness parameters
        len_wav_x = len(wav_x)
        if len_wav_x == 0:
            raise ValueError(
                "No samples remain after filtering. Cutoffs are too large for the "
                "available trace length. Reduce cutoffs or verify they are in the "
                f"same units as the X data ({self.x_units})."
            )
        self.R_params = {}
        self.R_params['Ra'] = (sum(map(abs, self.roughness[1]))
                            / len_wav_x, self.y_units)
        self.R_params['Rq'] = (np.sqrt(sum(map(np.square, self.roughness[1]))
                            / len_wav_x), self.y_units)
        self.R_params['Rsk'] = (sum([x ** 3 for x in self.roughness[1]]) /
                             ((self.R_params['Rq'][0] ** 3) * len_wav_x), "")
        self.R_params['Rku'] = (sum([x ** 4 for x in self.roughness[1]]) /
                             ((self.R_params['Rq'][0] ** 4) * len_wav_x), "")
        # ISO 4287 amplitude parameters (Rp/Rv on filtered roughness profile)
        Rp = float(np.max(self.roughness[1]))
        Rv = float(abs(np.min(self.roughness[1])))
        Rz = Rp + Rv
        self.R_params['Rp'] = (Rp, self.y_units)
        self.R_params['Rv'] = (Rv, self.y_units)
        self.R_params['Rz'] = (Rz, self.y_units)
        self.R_params['Rt'] = (Rz, self.y_units)
        
        if kwargs['PLOT_ROUGHNESS'] or kwargs['PLOT_ALL']:
            self.plot_roughness()

        if kwargs['PLOT_MR'] or kwargs['PLOT_ALL']:
            self.plot_material_ratio()

    def wear_track_depth(self):
        return min(self.primary[1]), "μm"

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
            a.set_xlabel(f"size, {self.x_units}")
            a.set_ylabel(f"height, {self.y_units}")
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
        plt.draw()

    @timeit
    def get_material_ratio(self, samples=1000, Pk_Offset=0.01, Vy_Offset=0.01):
        self.mr_params = {}
        self.Pk_Offset, self.Vy_Offset = Pk_Offset, Vy_Offset
        
        # sort the uniformly sampled profile in descending order
        self.material_ratio = np.sort(self.roughness[1])[::-1]
        self.material_ratio_all = self.roughness[:, self.roughness[1].argsort()[::-1]]
        size = self.roughness[0].size
        x = np.linspace(0, 100, size)
        self.material_ratio_all = np.vstack((self.material_ratio_all, x))

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
            if abs(m) > bf40_grad: break

        # y = mx + c
        self.bf40at0 = self.bf40_eq[1]
        self.bf40at100 = self.bf40_eq[0] * 100 + self.bf40at0
        self.mr_params['Rvkx'] = self.bf40at100 - self.material_ratio_all[1][-1] 
        self.mr_params['Rk'] = self.bf40at0 - self.bf40at100
        # intersection of self.roughness and best fit line
        self.mr_params['Rpkx'] = self.material_ratio_all[1][0] - self.bf40at0

        # calculate Rmrk and Rak params
        self.mr_params['Rak1'] = 0
        self.mr_params['Rak2'] = 0
        self.Rak1_points = [], []
        self.Rak2_points = [], []

        # iterate from bottom of profile up to bf line x=100 intercept
        for i, y in enumerate(self.material_ratio[1][::-1]):
            x = self.material_ratio[0][::-1][i]
            if y > self.bf40at100:
                self.mr_params['Rmrk2'] = x
                self.Rmrk2_y = y
                break

        # calculate Rak values
        count = 0
        self.mr_params['Rmrk1'] = 0
        self.Rmrk1_y = 0
        for i in range(self.material_ratio_all[1].size - 1):
            dt = self.material_ratio_all[2][i + 1] - \
                 self.material_ratio_all[2][i]
            x = self.material_ratio_all[0][i]   # position
            x_p = self.material_ratio_all[2][i] # percentage
            y = self.material_ratio_all[1][i]   # height
            y_1 = self.material_ratio_all[1][i + 1] # next height

            if y > self.bf40at0:
                self.Rak1_points[0].append(x)
                self.Rak1_points[1].append(y)
                self.mr_params['Rak1'] += dt * ((y + y_1) / 2 - self.bf40at0)
                count += 1
            else:
                if not self.mr_params['Rmrk1']:
                    self.mr_params['Rmrk1'] = x_p
                    self.Rmrk1_y = y
                self.Rak1_points[0].append(x)
                self.Rak1_points[1].append(np.nan)

        count = 0
        self.mr_params['Rmrk2'] = 0
        self.Rmrk2_y = 0
        for i in range(self.material_ratio_all[1].size - 1):
            dt = self.material_ratio_all[2][i + 1] - \
                 self.material_ratio_all[2][i]
            x = self.material_ratio_all[0][::-1][i]   # position
            x_p = self.material_ratio_all[2][::-1][i] # percentage
            y = self.material_ratio_all[1][::-1][i]   # height
            y_1 = self.material_ratio_all[1][::-1][i + 1] # next height

            if y < self.bf40at100:
                self.Rak2_points[0].append(x)
                self.Rak2_points[1].append(y)
                self.mr_params['Rak2'] -= dt * ((y + y_1) / 2 - self.bf40at100)
                count += 1
            else:
                if not self.mr_params['Rmrk2']:
                    self.mr_params['Rmrk2'] = x_p
                    self.Rmrk2_y = y
                self.Rak2_points[0].append(x)
                self.Rak2_points[1].append(np.nan)

        # for plotting plateau and dale regions on top of the roughness profile
        self.Rak1_points = np.asarray(self.Rak1_points)
        self.Rak2_points = np.asarray(self.Rak2_points)
        self.Rak1_points = self.Rak1_points[:, self.Rak1_points[0, :].argsort()]
        self.Rak2_points = self.Rak2_points[:, self.Rak2_points[0, :].argsort()]

        
        self.mr_params['Rpk'] = 2 * self.mr_params['Rak1'] / \
                                self.mr_params['Rmrk1']
        self.mr_params['Rvk'] = 2 * self.mr_params['Rak2'] / \
                                (100 - self.mr_params['Rmrk2'])

        print(f'num_data_points: {self.material_ratio_all[1].size}')
        print(f'Rak1: {self.mr_params["Rak1"]:.2f}, Rmrk1: {self.mr_params["Rmrk1"]:.2f}')
        print(f'Rak2: {self.mr_params["Rak2"]:.2f}, Rmrk2: {self.mr_params["Rmrk2"]:.2f}')
        print(f'bf40at100: {self.bf40at100:.2f}, bf40_eq: {self.bf40at0:.2f}')
        print(f'Rpk: {self.mr_params["Rpk"]:.2f} Rvk: {self.mr_params["Rvk"]:.2f}')

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
        
        # plot roughness + plateau and dale regions
        axs[0, 0].plot(*self.roughness, linewidth=0.5, color="blue")
        axs[0, 0].set_xlim(self.roughness[0][0], self.roughness[0][-1])
        axs[0, 0].plot(*self.Rak1_points, linewidth=0.75, color="green")
        axs[0, 0].plot(*self.Rak2_points, linewidth=0.75, color="green")

        
        #print(f'Rpk_line: {Rpk_line}, Rvk_line: {Rvk_line}')

        # plot material ratio
        axs[0, 1].plot(*self.material_ratio, color="red")
        axs[0, 1].set_ylim(0, 100)
        for xlabel_i in axs[0, 1].get_yticklabels():
            xlabel_i.set_visible(False)
        x = np.linspace(0, 100, 100)
        y = self.bf40_eq[0] * x + self.bf40at0
        axs[0, 1].plot(x, y, color="blue", linewidth=0.5)
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
            a.axhline(self.bf40at0, linestyle='--',
                      color="green", linewidth=0.5)
            a.axhline(self.bf40at100, linestyle='--',
                      color="green", linewidth=0.5)
            # plot Rpk and Rvk lines
            a.axhline(y=self.Rmrk1_y + self.mr_params['Rpk'], 
                      linestyle='dotted', color="k", linewidth=0.75)
            a.axhline(y=self.Rmrk2_y +-self.mr_params['Rvk'],
                      linestyle='dotted', color="k", linewidth=0.75)

        # create Rmr parameters table
        rows = ['Rk', 'Rpk', 'Rvk', 'Rmrk1', 'Rmrk2', 'Rak1', 'Rak2', 'Rpkx', 'Rvkx']
        units = ['μm', 'μm', 'μm', 'μm', 'μm', '%', '%', 'μm.%', 'μm.%']
        columns = ['ISO 21290-2:2021 Parameter', 'Value', 'Unit']
        n_rows = len(rows)
        
        cell_text = []
        for i in range(n_rows):
            cell_text.append([round(self.mr_params[rows[i]], 3), units[i]])
        mr_table = table(axs[1, 1], cellText=cell_text,
                                 rowLabels=rows,
                                 colLabels=columns,
                                 loc='top',
                                 fontsize=10)
        
        axs[0, 0].set_title(f"{os.path.split(self.raw_data)[1]} - Roughness")
        axs[0, 1].set_title("Material Ratio")
        plt.tight_layout()
        plt.draw()

    def compute_Rmr(self, Cref=5.0):
        """Material ratio at slicing level (peak_at_Cref% - Rz/4).

        Stores the result on ``self.R_params['Rmr']`` and saves the slicing
        level on ``self.Rmr_target_level`` for plotting.
        """
        rough = self.roughness[1]
        n = rough.size
        unit_label = f"% (@Rz/4, Cref={Cref:g}%)"
        if n == 0:
            self.R_params['Rmr'] = (0.0, unit_label)
            self.Rmr_target_level = 0.0
            return
        Rz = self.R_params['Rz'][0]
        sorted_desc = np.sort(rough)[::-1]
        idx = max(0, min(n - 1, int(round(Cref / 100.0 * n))))
        c0 = sorted_desc[idx]
        target = c0 - Rz / 4.0
        above = int(np.sum(sorted_desc >= target))
        Rmr = above / n * 100.0
        self.R_params['Rmr'] = (Rmr, unit_label)
        self.Rmr_target_level = target
        self.Rmr_Cref_level = c0

    def build_overview_figure(self, Cref=5.0):
        """Build a matplotlib ``Figure`` (no pyplot) suitable for GUI embedding.

        The figure contains four panels:
          * Raw data with the polynomial leveling fit overlaid
          * Roughness profile with the waviness curve overlaid (red)
          * Bearing ratio (Abbott) curve, depth on y-axis in y-units, % on x-axis
          * R-parameters table
        """
        from matplotlib.figure import Figure

        if not hasattr(self, 'material_ratio'):
            self.get_material_ratio()
        self.compute_Rmr(Cref=Cref)

        fig = Figure(figsize=(11, 7.5))
        gs = fig.add_gridspec(2, 2,
                              width_ratios=[2, 1],
                              hspace=0.40, wspace=0.30)
        ax_raw = fig.add_subplot(gs[0, 0])
        ax_bear = fig.add_subplot(gs[0, 1])
        ax_rough = fig.add_subplot(gs[1, 0])
        ax_table = fig.add_subplot(gs[1, 1])

        # --- Raw data + leveling fit -------------------------------------
        ax_raw.plot(self.raw_data_xy[0], self.raw_data_xy[1],
                    lw=0.5, color='steelblue', label='raw')
        if self.level_fit_y is not None:
            ax_raw.plot(self.level_fit_x, self.level_fit_y,
                        color='red', lw=1.0,
                        label=f'order {self.order} fit')
            ax_raw.legend(fontsize=8, loc='best')
        ax_raw.set_title('Raw Data + Leveling Fit')
        ax_raw.set_xlabel(f'size, {self.x_units}')
        ax_raw.set_ylabel(f'height, {self.y_units}')
        ax_raw.grid(True, ls=':', alpha=0.5)

        # --- Roughness + waviness ---------------------------------------
        ax_rough.plot(*self.roughness, lw=0.5, color='green', label='roughness')
        ax_rough.plot(*self.waviness, lw=0.9, color='red', label='waviness')
        ax_rough.set_title(
            f'Roughness + Waviness  (\u03bbc={self.long_cutoff}{self.x_units}, '
            f'\u03bbs={self.short_cutoff}{self.x_units})'
        )
        ax_rough.set_xlabel(f'size, {self.x_units}')
        ax_rough.set_ylabel(f'height, {self.y_units}')
        ax_rough.axhline(0, color='black', lw=0.5)
        ax_rough.legend(fontsize=8, loc='best')
        ax_rough.grid(True, ls=':', alpha=0.5)

        # --- Bearing ratio curve ----------------------------------------
        # self.material_ratio[0] is %; self.material_ratio[1] is depth
        ax_bear.plot(self.material_ratio[0], self.material_ratio[1],
                     color='blue', lw=1.0)
        ax_bear.set_title('Bearing Ratio Curve')
        ax_bear.set_xlabel('Material Ratio %')
        ax_bear.set_ylabel(f'depth, {self.y_units}')
        ax_bear.set_xlim(0, 100)
        ax_bear.grid(True, ls=':', alpha=0.5)
        if hasattr(self, 'Rmr_target_level'):
            ax_bear.axhline(self.Rmr_target_level, color='gray',
                            ls='--', lw=0.7,
                            label=f'Rz/4 slice')
            Rmr_val = self.R_params['Rmr'][0]
            ax_bear.axvline(Rmr_val, color='magenta', ls=':', lw=0.8,
                            label=f'Rmr = {Rmr_val:.2f}%')
            ax_bear.legend(fontsize=7, loc='best')

        # --- R-parameter table -------------------------------------------
        ax_table.set_axis_off()
        rows = ['Ra', 'Rp', 'Rt', 'Rv', 'Rz', 'Rq', 'Rsk', 'Rmr']
        cells = []
        for r in rows:
            v, u = self.R_params.get(r, ('-', ''))
            if isinstance(v, (int, float)):
                cells.append([f'{v:.4f}', u])
            else:
                cells.append([str(v), u])
        tbl = ax_table.table(cellText=cells,
                             rowLabels=rows,
                             colLabels=['Value', 'Unit'],
                             loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)
        ax_table.set_title(f'R-Parameters  (Cref = {Cref:g}%)', fontsize=10)

        fig.suptitle(os.path.basename(self.raw_data), fontsize=11)
        return fig

    def __str__(self):
        p = f'\nProcessed {self.raw_data} with λc = {self.long_cutoff}mm\n' \
            f'and λc/s = {self.short_cutoff*1000}μm\n\nParam\tValue'
        for key in self.R_params:
            p += f"\n{key}:\t{self.R_params[key][0]:.3f}{self.R_params[key][1]}"
        p += "\n"
        return p


if __name__ == "__main__":
    data = "example/example_trace.txt"
    short_cutoff = 2.5 / 1000
    long_cutoff = 0.8
    surface_texture = SurfaceTexture(data, short_cutoff, long_cutoff,
                                     order=1)
    #surface_texture.plot_material_ratio()
    surface_texture.plot_roughness()
    surface_texture.plot_material_ratio()
    print(surface_texture.wear_track_depth())
    

    plt.show()

