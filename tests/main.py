import math
import numpy as np
import xarray as xr
from pyspammodel._aero_spam import AeroSpam
from pyspammodel._solar_spam import SolarSpam

f107_int = 100
f107_float = 100.
f107_single = [100]
f107_list = [100, 155, 155]
f107_nparray = np.array([100, 155, 155])
f107_wrongtype = ((100,150), (200,300))

e = AeroSpam()
print(e.get_spectra(150))

# p = xr.open_dataset('F:\MainProjects\InProgress\pyspammodel\src\pyspammodel\_coeffs\_solar_spam_coeffs.nc').to_pandas()
# p = p.drop(0)
# x = float(-4.6505577e-5)
# p.loc[7, 'P3'] = x
# p.to_csv('data_miss05.csv', index=False)
# print(p.iloc[7])


#---------------Aero spam------------
# import pyspammodel as spam
# example = spam.AeroSpam()
# spectra = example.get_spectra(155.)
# print(spectra[0])

#------------Solar spam--------------
# import pyspammodel as spam
# example = spam.SolarSpam()
# spectrum = example.get_spectral_bands([155., 200.])
# print(spectrum['euv_flux_spectra'])