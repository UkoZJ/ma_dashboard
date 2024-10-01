# %% Imports
import geopandas as gpd
import h3
import h3.api.numpy_int
import h3.unstable.vect
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xesmf as xe
import xvec

import utility as ut
from utility import mask_from_vect

# %load_ext autoreload
# %autoreload 2


# %% Main functional forms
def Briere_func(cte, tmin, tmax, temp):
    outp = temp * cte * (temp - tmin) * (tmax - temp) ** 0.5
    return max(outp, 0) if not np.isnan(outp) else 0


def Quad_func(cte, tmin, tmax, temp):
    outp = -cte * (temp - tmin) * (temp - tmax)
    return max(outp, 0) if not np.isnan(outp) else 0


def QuadN_func(cte, c1, c2, temp):
    outp = cte * temp**2 + c1 * temp + c2
    return max(outp, 0) if not np.isnan(outp) else 0


# Function to compute the hatching rate
def h_f(hum, rain):
    erat = 0.5
    e0 = 1.5
    evar = 0.05
    eopt = 8.0
    efac = 0.01
    edens = 0.01

    hatch = (1 - erat) * (
        ((1 + e0) * np.exp(-evar * (rain - eopt) ** 2))
        / (np.exp(-evar * (rain - eopt) ** 2) + e0)
    ) + erat * (edens / (edens + np.exp(-efac * hum)))
    return hatch


# Parameters for Aedes albopictus and Aedes aegypti
params = {
    "albopictus": {
        "a_f": (Briere_func, 0.000193, 10.25, 38.32),
        "TFD_f": (Briere_func, 0.0488, 8.02, 35.65),
        "pLA_f": (Quad_func, 0.002663, 6.668, 38.92),
        "MDR_f": (Briere_func, 0.0000638, 8.6, 39.66),
        "lf_f": (Quad_func, 1.43, 13.41, 31.51),
        "dE_f": (Quad_func, 0.00071, 1.73, 40.51),
        "deltaE_f": (QuadN_func, 0.0019328, -0.091868, 1.3338874),
    },
    "aegypti": {
        "a_f": (Briere_func, 0.000202, 13.35, 40.08),
        "TFD_f": (Briere_func, 0.00856, 14.58, 34.61),
        "pLA_f": (Quad_func, 0.004186, 9.373, 40.26),
        "MDR_f": (Briere_func, 0.0000786, 11.36, 39.17),
        "lf_f": (Quad_func, 0.148, 9.16, 37.73),
        "dE_f": (Briere_func, 0.0003775, 14.88, 37.42),
        "deltaE_f": (QuadN_func, 0.004475, -0.210787, 2.55237),
    },
}


def compute_param(func, *args):
    return func(*args)


# Generic RM function
def R0_func(species, Te, rain, hum):
    if np.isnan(Te) or np.isnan(rain) or np.isnan(hum):
        return np.nan

    species_params = params[species]

    a = compute_param(*species_params["a_f"], Te)
    f = (1.0 / 2.0) * compute_param(*species_params["TFD_f"], Te)
    deltaa = compute_param(*species_params["lf_f"], Te)
    dE = compute_param(*species_params["dE_f"], Te)
    probla = compute_param(*species_params["pLA_f"], Te)
    h = h_f(hum, rain)
    deltaE = compute_param(*species_params["deltaE_f"], Te)

    R0 = ((f * a * deltaa) * probla * ((h * dE) / (h * dE + deltaE))) ** (1.0 / 3.0)
    return R0


# %% Example usage of R0_func

# Define temperature, rainfall, and human density
temperature = 25  # degrees Celsius
rainfall = 10  # mm
human_density = 1000  # people per square km

# Compute R0 for Aedes albopictus
R0_albopictus = R0_func("albopictus", temperature, rainfall, human_density)
print(f"Suitability index for Aedes albopictus: {R0_albopictus}")

# Compute R0 for Aedes aegypti
R0_aegypti = R0_func("aegypti", temperature, rainfall, human_density)
print(f"Suitability index for Aedes aegypti: {R0_aegypti}")


# %% Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Smooth function P
def decision_index(MSI, RD, r, k):
    S_MSI = sigmoid(k * (MSI - 0.5))
    S_RD = sigmoid(-k * (RD - r))
    P = S_MSI * S_RD + 0.5 * (1 - S_MSI) * S_RD
    return P


# Parameters
r = 0.5
k = 30
# Generate a grid of MSI and RD values
MSI_values = np.linspace(0, 1, 100)
RD_values = np.linspace(0, 1, 100)
MSI_grid, RD_grid = np.meshgrid(MSI_values, RD_values)

# Compute P for each pair of MSI and RD
P_grid = decision_index(MSI_grid, RD_grid, r, k)

# Plotting the results
plt.figure(figsize=(10, 8))
cp = plt.contourf(MSI_grid, RD_grid, P_grid, levels=50, cmap="viridis")
plt.colorbar(cp)
plt.xlabel("MSI")
plt.ylabel("RD")
plt.title("Smooth Decision Index P")
plt.show()

# %% Load and process data
ds = xr.load_dataset("../data/01_raw/era5/temperature_2m_t_2023.nc")
lon_ds = ds.coords["longitude"].values
lat_ds = ds.coords["latitude"].values
mask = xr.load_dataset("../data/01_raw/era5/mask_land.nc")
mask.coords["longitude"] = mask.coords["longitude"].values - 180

# %% Plot masked temperature
(mask.mask_land * ds.t2m[0]).plot()

# %% H3 indexing
resolution = 3

lon, lat = xr.broadcast(ds.longitude, ds.latitude)

index = h3.unstable.vect.geo_to_h3(lat.data.ravel(), lon.data.ravel(), resolution)
index.shape = lon.shape

len(np.unique(index)) / lon.size

# %% Add H3 index to dataset
ds.coords["index"] = ("latitude", "longitude"), index.transpose()
ds

# %% Plot H3 index
ds.index.plot()

# %% Define bounding box
lon_min, lon_max = ds.longitude.min().values.item(), ds.longitude.max().values.item()
lat_min, lat_max = ds.latitude.min().values.item(), ds.latitude.max().values.item()

bbox_coords = [
    (lon_min - 180, lat_min),
    (lon_min - 180, lat_max),
    (lon_max - 180, lat_max),
    (lon_max - 180, lat_min),
    (lon_min - 180, lat_min),
]
bbox = shapely.Polygon(bbox_coords)

# %% H3 polyfill
bbox_coords_lat_first = [(lat, lon) for lon, lat in bbox_coords]
bbox_indexes = np.array(
    list(h3.api.basic_int.polyfill_polygon(bbox_coords_lat_first, resolution))
)

ll_points = np.array([h3.api.numpy_int.h3_to_geo(i) for i in bbox_indexes])
ll_points_lon_first = ll_points[:, ::-1]

# %% Interpolate data to H3 grid
coords = {"cell": bbox_indexes}

dsi = ds.interp(
    longitude=xr.DataArray(ll_points_lon_first[:, 0], dims="cell", coords=coords),
    latitude=xr.DataArray(ll_points_lon_first[:, 1], dims="cell", coords=coords),
)

dsi2 = dsi.drop_vars(["longitude", "latitude", "index"])
dsi2.cell.attrs = {"grid_name": "h3", "resolution": resolution}

# %% Plot interpolated data
ds.t2m[0].plot()
plt.scatter(
    ll_points_lon_first[:, 0] - 180,
    ll_points_lon_first[:, 1] + 90,
    cmap="cool",
    edgecolor="black",
)

# %% Regrid data
ds = xr.open_dataset("../data/01_raw/era5/temperature_2m_p01_t_2023.nc")

ds_target_grid = xr.Dataset(
    {
        "lat": (["latitude"], np.arange(-90, 90.25, 0.25), {"units": "degrees_north"}),
        "lon": (["longitude"], np.arange(-180, 180, 0.25), {"units": "degrees_east"}),
    }
)

regridder = xe.Regridder(ds, ds_target_grid, "nearest_s2d", periodic=True)
ds_regridded = regridder(ds["t2m"], keep_attrs=True)

# %% Compare regridded data
ds_p25 = xr.open_dataset("../data/01_raw/era5/temperature_2m_t_2023.nc")
(ds_p25.isel(time=0)["t2m"] - ds_regridded.isel(time=0)).plot()

# %% Create land-sea mask
mask_da = mask_from_vect(
    vect_path="/home/uko/Dev/research_datasets/costline/costline_buffer.gpkg",
    ref_path="../data/01_raw/era5/temperature_2m_t_2023.nc",
)
mask_da.to_netcdf("./data/land_sea_mask.nc")

# %% Load and process data
ds = xr.load_dataset("../data/01_raw/era5/temperature_2m_t_2023.nc")
reports = pd.read_parquet("../data/02_intermediate/reports.parquet")
land_mask = gpd.GeoDataFrame(
    geometry=gpd.read_file("../data/01_raw/costline/costline_buffer.gpkg").geometry,
    crs="EPSG:4326",
)
regs = gpd.read_file("../data/01_raw/costline/ne_50m_coastline.json")

# %% Create grid and count points
cell_size = 0.1
land_mask_buffer = gpd.GeoDataFrame(
    geometry=land_mask.to_crs("epsg:3857").buffer(distance=cell_size / 2).to_crs(4326)
)

grid = gpd.read_parquet("../data/01_raw/era5/grid.parquet")

points_ct = ut.count_points_on_grid(reports[["lat", "lon"]], grid)
grid["point_ct"] = points_ct
grid["point_density"] = grid["point_ct"] / (grid["area"] / 1e6)

# %% Plot point density
fig, ax = plt.subplots()
grid.cx[-10:5, 35:45].plot(
    column="point_density",
    missing_kwds={"color": "lightgrey"},
    norm=mpl.colors.LogNorm(),
    legend=True,
    ax=ax,
)
regs.plot(edgecolor="k", facecolor="none", ax=ax)
land_mask_buffer.plot(edgecolor="gray", facecolor="none", ax=ax)

ax.set_ylim(35, 45)
ax.set_xlim(-10, 5)

# %% Extract points from dataset
gs = gpd.GeoSeries(grid["centroids"], crs=4326)
extracted = ds.xvec.extract_points(gs, x_coords="longitude", y_coords="latitude")
gdf = extracted["t2m"].isel(time=0).xvec.to_geodataframe()

# %%
gdf
# %%
