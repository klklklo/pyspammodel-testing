import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

# SolarSPAM model for testing.
# All functionality is preserved, the ability to change the coefficient table is added.
class SolarSpam:
    '''
    Solar-SPAM model class.
    '''
    def __init__(self, file):
        self._dataset = pd.read_csv(file)
        self._coeffs = np.vstack((np.array(self._dataset['P1'], dtype=np.float64),
                                  np.array(self._dataset['P2'], dtype=np.float64),
                                  np.array(self._dataset['P3'], dtype=np.float64))).transpose()

    def _get_f107(self, f107):
        '''
        Method for creating the daily F10.7 index matrix that will be used to calculate the spectrum.
        Returns a matrix with rows [F10.7 ^ 2; F10.7; 1] for each passed value F10.7.
        :param f107: single value of the daily index F10.7 or an array of such values.
        :return: numpy array for model calculation.
        '''
        try:
            if isinstance(f107, float) or isinstance(f107, int):
                return np.array([f107 ** 2, f107, 1], dtype=np.float64).reshape(1, 3)
            return np.vstack([np.array([x ** 2, x, 1]) for x in f107], dtype=np.float64)
        except TypeError:
            raise TypeError('Only int, float or array-like object types are allowed')

    def get_spectral_bands(self, f107):
        '''
        Model calculation method. Returns the xarray dataset values of radiation fluxes in all intervals
        of the spectrum of the interval 0-190 nm
        :param f107: single value of the daily index F10.7 or an array of such values
        :return: xarray Dataset [euv_flux_spectra, line_lambda]
        '''
        f107 = self._get_f107(f107)
        res = np.dot(self._coeffs, f107.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'F107'), res),
                                     'line_lambda': ('band_number', self._dataset['lambda'].values)},
                          coords={'band_center': self._dataset['lambda'].values,
                                  'F107': f107[:, 1],
                                  'band_number': np.arange(190)})

    def get_spectra(self, f107):
        '''
        Model calculation method. Used to unify the interface with AeroSpam class.
        :param f107: single value of the daily index F10.7 or an array of such values.
        :return: xarray Dataset [euv_flux_spectra, line_lambda]
        '''
        return self.get_spectral_bands(f107)


# File upload with coefficient table
#------------------
s = SolarSpam('table_A1_corrected.csv')
f107 = 195.
#------------------

# Reading Octave octave_data
#------------------
with open(f'octave_data/data_{int(f107)}.txt', 'r') as f:
    octave_data = f.read()

octave_data = np.array([float(i) for i in octave_data.split()]).reshape(190, 1)
#------------------

# Spectrum calculation by SolarSPAM model
#------------------
res = s.get_spectral_bands(f107)['euv_flux_spectra'].values
diff = abs(res - octave_data)
#------------------

# Plotting spectra
#------------------
fig, axs = plt.subplots(4,1)
plt.subplots_adjust(wspace=0.3, hspace=0.6)
axs[0].semilogy(res, label='Python implementation', drawstyle='steps')
axs[0].set_title('Python implementation')
axs[0].set_xlim(0,189)
axs[0].set_xticks(np.arange(0,191,10))
axs[0].set_xlabel('Wavelength, nm')
axs[0].set_ylabel('F, W m^-2 nm^-1', fontsize=10)

axs[1].semilogy(octave_data, color='orange', drawstyle='steps')
axs[1].set_title('Matlab implementation (original)')
axs[1].set_xlim(0,189)
axs[1].set_xticks(np.arange(0,191,10))
axs[1].set_xlabel('Wavelength, nm')
axs[1].set_ylabel('F, W m^-2 nm^-1', fontsize=10)

axs[2].semilogy(octave_data, label='Matlab implementation', color='orange', drawstyle='steps')
axs[2].semilogy(res, label='Python implementation', drawstyle='steps')
axs[2].set_title('Overlay')
axs[2].legend()
axs[2].set_xlim(0,189)
axs[2].set_xticks(np.arange(0,191,10))
axs[2].set_xlabel('Wavelength, nm')
axs[2].set_ylabel('F, W m^-2 nm^-1', fontsize=10)

axs[3].plot(diff, drawstyle='steps')
axs[3].set_xlim(0,189)
axs[3].set_xticks(np.arange(0,191,10))
axs[3].set_xlabel('Wavelength, nm')
axs[3].set_ylabel('Difference, W m^-2 nm^-1', fontsize=10)

plt.show()
#------------------
