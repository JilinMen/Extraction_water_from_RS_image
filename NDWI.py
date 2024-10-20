# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:06:37 2024

extract water area from array

@author: jmen
"""
import h5py
import os
import shutil
from osgeo import gdal, osr
import numpy as np
from tif2shp import tif2shp
    
def NDWI(G, SWIR, threshold=0):
    
    return ((G-SWIR)/(G+SWIR)<threshold).astype(int)

def Speckle_removal(array, remove_pixels=100, neighbours=8):
    from scipy.ndimage import label, find_objects
    # Label connected regions
    structure = np.ones((3, 3)) if neighbours == 8 else np.eye(3)
    labeled_array, num_features = label(array, structure=structure)

    # Find objects (connected regions) in the labeled array
    objects = find_objects(labeled_array)

    # Remove small objects (speckles)
    for i, obj_slice in enumerate(objects):
        # Get the region of interest
        region = labeled_array[obj_slice] == i + 1

        # If the region is smaller than the threshold, set it to 0 (remove it)
        if np.sum(region) < remove_pixels:
            array[obj_slice][region] = 0

    return array

def save_tif(image, out_path, lat, lon):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    
    if image.ndim == 2:
        image = image[np.newaxis, :, :]  # 添加一个波段维度
    elif image.ndim != 3:
        raise ValueError("输入图像数组必须是 2D 或 3D")
        
    channels, rows, cols = image.shape
    
    # Calculate pixel size
    pixel_width = (lon_max - lon_min) / (cols - 1)
    pixel_height = (lat_max - lat_min) / (rows - 1)
    
    # Create the transformation
    transform = from_origin(lon_min - pixel_width/2, lat_max + pixel_height/2, pixel_width, pixel_height)
    
    # Define the CRS (WGS84)
    crs = CRS.from_epsg(4326)
    
    # Create the GeoTIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=channels,
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        # Write the data
        for i in range(channels):
            dst.write(image[i, :, :], i+1)  # rasterio 波段索引从 1 开始

    print(f"GeoTIFF saved to {output_path}")

if __name__=='__main__':
    input_path = r'H:\Satellite_processing_ERSL\L2\LC09_L2TP_021037_20240608_20240608_02_T1\L9_OLI_2024_06_08_16_24_27_021037_L2R.nc'
    output_path = input_path.split('.')[0]+'.tif'
    
    ds = h5py.File(input_path,'r')
    
    G = np.array(ds['rhos_561'])
    SWIR = np.array(ds['rhos_1608'])
    
    lat = np.array(ds['lat'])
    lon = np.array(ds['lon'])
    
    water_mask = NDWI(G, SWIR)
    
    water_mask = Speckle_removal(water_mask)
    
    save_tif(water_mask, output_path, lat, lon)
    
    output_shp_path = input_path.split('.')[0]+'.shp'
    tif2shp(output_path, output_shp_path)