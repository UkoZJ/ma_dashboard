import os
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from affine import Affine
from geopandas.tools import sjoin
from pyproj import Geod
from rasterio import features
from scipy import interpolate
from shapely.geometry import MultiLineString, Point, Polygon


def mask_from_vect(vect_path, ref_path, set_value: int = 1):
    # Load the polygon shapefile
    polygon = gpd.read_file(vect_path)
    # Load the reference datacube
    ds = xr.load_dataset(ref_path)
    longitudes = ds.coords["longitude"].values
    latitudes = ds.coords["latitude"].values

    # Ensure the polygon is in a geographic coordinate system (e.g., EPSG:4326)
    if polygon.crs.to_epsg() != 4326:
        polygon = polygon.to_crs(epsg=4326)

    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = (
        longitudes.min(),
        latitudes.min(),
        longitudes.max(),
        latitudes.max(),
    )

    # Calculate the number of rows and columns
    width = len(longitudes)
    height = len(latitudes)
    cell_size = np.abs(longitudes[0] - longitudes[1])

    # Create an affine transform
    transform = Affine.translation(minx, maxy) * Affine.scale(cell_size, -cell_size)

    # Create a raster with the same dimensions
    raster = np.zeros((height, width), dtype=np.uint8)

    # Rasterize the polygon
    shapes = ((geom, 1) for geom in polygon.geometry)
    rasterized = features.rasterize(
        shapes=shapes, out=raster, transform=transform, all_touched=True
    )

    mask = (rasterized == set_value).astype(np.uint8)

    mask_da = xr.DataArray(
        mask,
        coords=[("latitude", latitudes), ("longitude", longitudes)],
        name="land_sea_mask",
    )
    # Set georeference attributes
    mask_da = mask_da.rio.write_crs("EPSG:4326")
    mask_da.rio.write_transform(transform)

    return mask_da


def interpolate_mask(da, mask):
    # Convert to numpy array
    data = da.values

    # Get coordinates
    y, x = np.indices(data.shape)

    # Interpolate
    data[mask] = interpolate.NearestNDInterpolator((y[mask], x[mask]), data[mask])

    # Create new DataArray with interpolated values
    new_da = xr.DataArray(data, coords=da.coords, dims=da.dims, attrs=da.attrs)

    return new_da


def make_grid(
    lat_range=(-90, 90),
    lon_range=(-180, 180.0),
    cell_size=0.25,
    vec_mask: gpd.GeoDataFrame = None,
    save_grid: Optional[str] = None,
) -> gpd.GeoDataFrame:
    lats = np.arange(lat_range[0], lat_range[1], cell_size)
    lons = np.arange(lon_range[0], lon_range[1], cell_size)
    corners = np.array(np.meshgrid(lons, lats)).T.reshape(-1, 2)
    centroids = corners + cell_size / 2
    centroids = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(centroids[:, 0], centroids[:, 1]),
        crs="EPSG:4326",
    )

    def create_polygon(ci, cell_size):
        return Polygon(
            [
                (ci[0], ci[1]),
                (ci[0] + cell_size, ci[1]),
                (ci[0] + cell_size, ci[1] + cell_size),
                (ci[0], ci[1] + cell_size),
            ]
        )

    if vec_mask is not None:
        within_centroinds = gpd.sjoin(
            centroids, vec_mask, how="inner", predicate="within"
        ).drop("index_right", axis=1)
        corners = corners[within_centroinds.index]
    # Generate polygons for the grid cells
    polygons = [create_polygon(ci, cell_size) for ci in corners]

    # Create GeoDataFrame
    if vec_mask is not None:
        grid = gpd.GeoDataFrame(index=within_centroinds.index, geometry=polygons)
    else:
        grid = gpd.GeoDataFrame(geometry=polygons)
    grid.set_crs(epsg="4326", inplace=True)
    grid["centroids"] = centroids
    grid["area"] = grid.to_crs("ESRI:53034").geometry.area
    grid.index.name = "id_cell"
    if save_grid is not None:
        grid.to_parquet(save_grid, index=True)
    return grid


def count_points_on_grid(df_points, grid) -> gpd.GeoSeries:
    geometry = gpd.points_from_xy(df_points.lon, df_points.lat)
    points = gpd.GeoDataFrame(df_points, geometry=geometry, crs="4326")
    points.index.name = "id_point"
    points_ct = sjoin(points, grid, how="left").groupby(["id_cell"]).size()
    points_ct.index = points_ct.index.astype("int64")

    return points_ct
