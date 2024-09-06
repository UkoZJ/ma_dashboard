import numpy as np
import xarray as xr
from typing import List

def _briere(cte, tmin, tmax, temp):
    dt_max = tmax - temp
    dt_max = np.where(dt_max<0, 0, dt_max)
    outp = temp * cte * (temp - tmin) * np.sqrt(dt_max)
    return np.where(outp < 0, 0, outp)

def _quad(cte, tmin, tmax, temp):
    outp = -cte * (temp - tmin) * (temp - tmax)
    return np.where(outp < 0, 0, outp)

def _quadn(cte, c1, c2, temp):
    outp = cte * temp**2 + c1 * temp + c2
    return np.where(outp < 0, 0, outp)

# Function to compute the hatching rate
def h_f(hum, rain):
    # Constants
    erat = 0.5
    e0 = 1.5
    evar = 0.05
    eopt = 8
    efac = 0.01
    edens = 0.01
    
    hatch = (1-erat)*(((1+e0)*np.exp(-evar*(rain-eopt)**2)) / (np.exp(-evar*(rain - eopt)**2) + e0)) + erat*(edens / (edens + np.exp(-efac*hum)))
    return hatch

def r0_albopictus(Te, rain, hum):
    """Function to compute R0 for Aedes albopictus"""
    def a_f(temp):
        return _briere(0.000193, 10.25, 38.32, temp)

    def tfd_f(temp):
        return _briere(0.0488, 8.02, 35.65, temp)

    def pla_f(temp):
        return _quad(0.002663, 6.668, 38.92, temp)

    def mdr_f(temp):
        return _briere(0.0000638, 8.6, 39.66, temp)

    def lf_f(temp):
        return _quad(1.43, 13.41, 31.51, temp)

    def de_f(temp):
        return _quad(0.00071, 1.73, 40.51, temp)

    def deltae_f(temp):
        return _quadn(0.0019328, -0.091868, 1.3338874, temp)

    a = a_f(Te)
    f = (1/2) * tfd_f(Te)
    deltaa = lf_f(Te)
    dE = de_f(Te)
    probla = pla_f(Te)
    h = h_f(hum, rain)
    deltaE = deltae_f(Te)
    
    R0 = ((f * a * deltaa) * probla * ((h * dE) / (h * dE + deltaE)))**(1/3)
    return R0

def r0_aegypti(Te, rain, hum):
    """Function to compute R0 for Aedes aegypti"""

    def a_f(temp):
        return _briere(0.000202, 13.35, 40.08, temp)

    def efd_f(temp):
        return _briere(0.00856, 14.58, 34.61, temp)

    def pla_f(temp):
        return _quad(0.004186, 9.373, 40.26, temp)

    def mdr_f(temp):
        return _briere(0.0000786, 11.36, 39.17, temp)

    def lf_f(temp):
        return _quad(0.148, 9.16, 37.73, temp)

    def de_f(temp):
        return _briere(0.0003775, 14.88, 37.42, temp)

    def deltae_f(temp):
        return _quadn(0.004475, -0.210787, 2.552370, temp)
    
    a = 1
    f = efd_f(Te)
    deltaa = lf_f(Te)
    dE = de_f(Te)
    probla = pla_f(Te)
    h = h_f(hum, rain)
    deltaE = deltae_f(Te)
    
    R0 = ((f * a * deltaa) * probla * ((h * dE) / (h * dE + deltaE)))**(1/3)
    return R0

def r0(
        temperature:xr.Dataset, # Celsius
        rainfall:xr.Dataset, # mm
        human_density:xr.Dataset, # people per km2
        species: List[str] = ['albopictus', 'aegypti']
        ):

    r0_func = {'albopictus':r0_albopictus, 'aegypti':r0_aegypti}

    if 'time' in human_density.dims:
        time_dim = temperature['time']
        human_density = human_density.expand_dims({'time': time_dim}, axis=0)
        human_density = human_density.assign_coords(time=time_dim)

    da = []
    for name in species:
        tmp_r0 = r0_func[name](temperature, rainfall, human_density)
        tmp_r0.name=f"r0_{name}"
        da.append(tmp_r0)

    return xr.combine_by_coords(da)

def mwi(ds:xr.Dataset):
    """
    Compute the mosquito wether index (MWI) given a dataset of wind components,
    air temperature, and dewpoint temperature.
    """

    windspeed_mps = np.sqrt(ds.u10**2 + ds.v10**2)
    temp_2m_c = ds.t2m - 273.15
    dewpoint_2m_c = ds.d2m - 273.15

    m = 7.59138
    tn = 240.726
    relative_humidity = 100*10**(m*( (dewpoint_2m_c/(dewpoint_2m_c + tn)) \
        - (temp_2m_c/(temp_2m_c+tn)) ))

    fw = xr.where(windspeed_mps <= 6, 1, 0)

    fh = xr.where(
        (relative_humidity >= 40) & (relative_humidity <= 95),
        (relative_humidity/55)-(40/55),
        0)

    # Multi if-else that works with xarrays
    ft = xr.where((temp_2m_c>15) & (temp_2m_c <=20), (.2*temp_2m_c)-3,
         xr.where((temp_2m_c>20) & (temp_2m_c<=25), 1,
         xr.where((temp_2m_c>25) & (temp_2m_c <= 30), (-.2*temp_2m_c)+6,
         0)))

    da_mwi = fw*fh*ft
    da_mwi.name = 'mwi'

    return da_mwi

def mwi_masked(mwi:xr.DataArray, mask:xr.DataArray):

    mask.coords['longitude'] = (mask.coords['longitude'] + 180) % 360 - 180
    mask = mask.sortby(mask['longitude'])
    mask_ = np.where(mask['mask_land']==False, np.nan, 1)
    return mask_* mwi

def binary_agg(x:xr.DataArray, thr:float):
    """Binary aggregation over time by threshold value"""
    da = xr.where(x <= thr, 0, xr.where(x > thr, 1, x))
    return da.sum('time', skipna=False) 