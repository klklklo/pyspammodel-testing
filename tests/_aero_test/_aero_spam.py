import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import models._misc as _m
import pandas as pd


class AeroSpam:

    calc = ''
    def __init__(self):
        self.dataset = pd.read_csv('_aero_coeffs.csv')
        self._coeffs = np.vstack((np.array(self.dataset['p1'], dtype=np.float64),
                                        np.array(self.dataset['p2'], dtype=np.float64),
                                        np.array(self.dataset['p3'], dtype=np.float64))).transpose()
    def _get_f107(self, f107):
        if isinstance(f107, float):
            return np.array([f107 ** 2, f107, 1], dtype=np.float64)[None, :]
        return np.vstack([np.array([x ** 2, x, 1]) for x in f107], dtype=np.float64)

    def get_spectral_bands_2(self, f107):

        with open(f'data/data_{int(f107)}.txt', 'r') as f:
            data = f.read()
        data = np.array([float(i) for i in data.split()]).reshape(37, 1)

        x = self._get_f107(f107)
        res = np.dot(self._coeffs, x.T)

        diff = abs(res - data)

        fig, axs = plt.subplots(4, 1)
        plt.subplots_adjust(wspace=0.3, hspace=0.8)
        axs[0].semilogy(res, label='Python implementation', drawstyle='steps')
        axs[0].set_title('Python implementation')
        axs[0].set_xlim(0, 36)
        axs[0].set_xticks(np.arange(0,37), self.dataset['center'], rotation=45)
        axs[0].set_xlabel('Wavelength, nm')
        axs[0].set_ylabel('F, W m^-2 nm^-1', fontsize=10)

        axs[1].semilogy(data, color='orange', drawstyle='steps')
        axs[1].set_title('Matlab implementation (original)')
        axs[1].set_xlim(0, 36)
        axs[1].set_xticks(np.arange(0,37), self.dataset['center'], rotation=45)
        axs[1].set_xlabel('Wavelength, nm')
        axs[1].set_ylabel('F, W m^-2 nm^-1', fontsize=10)

        axs[2].semilogy(data, label='Matlab implementation', color='orange', drawstyle='steps')
        axs[2].semilogy(res, label='Python implementation', drawstyle='steps')
        axs[2].set_title('Overlay')
        axs[2].legend()
        axs[2].set_xlim(0, 36)
        axs[2].set_xticks(np.arange(0,37), self.dataset['center'], rotation=45)
        axs[2].set_xlabel('Wavelength, nm')
        axs[2].set_ylabel('F, W m^-2 nm^-1', fontsize=10)

        axs[3].plot(diff, drawstyle='steps')
        axs[3].set_xlim(0, 36)
        axs[3].set_xticks(np.arange(0, 37), self.dataset['center'], rotation=45)
        axs[3].set_xlabel('Wavelength, nm')
        axs[3].set_ylabel('Difference, W m^-2 nm^-1', fontsize=10)

        plt.show()


f107 = 195.
a = AeroSpam()
a.get_spectral_bands_2(f107)
