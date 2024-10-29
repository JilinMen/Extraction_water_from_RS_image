# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:06:37 2024

extract water area from .nc file

@author: Jilin Men
"""
import h5py
import os
import shutil
from osgeo import gdal, osr
import numpy as np
from tif2shp import tif2shp
    
def NDWI(G, SWIR, threshold=0):
    
    return ((G-SWIR)/(G+SWIR)>threshold).astype(int)

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

def save_tif(image, output_path, lat, lon):
    """
    Save image data as a georeferenced TIFF file.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D or 3D image array (bands, height, width)
    output_path : str
        Path where the GeoTIFF will be saved
    lat : numpy.ndarray
        2D array of latitude values
    lon : numpy.ndarray
        2D array of longitude values
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import numpy as np
    
    # Input validation
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim != 3:
        raise ValueError("Image must be 2D or 3D array")
        
    channels, rows, cols = image.shape
    
    # Ensure lat/lon are 2D arrays
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("Latitude and longitude arrays must be 2D")
    
    # Get the exact bounds
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    
    # Calculate transform using bounds
    # from_bounds uses the exact corner coordinates without need for pixel size adjustments
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)
    
    # Define the CRS (WGS84)
    crs = CRS.from_epsg(4326)
    
    # Metadata for the GeoTIFF
    metadata = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': channels,
        'dtype': image.dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',  # Add compression
        'tiled': True,      # Enable tiling for better performance
        'interleave': 'band'
    }
    
    try:
        with rasterio.open(output_path, 'w', **metadata) as dst:
            # Write the data
            for i in range(channels):
                dst.write(image[i, :, :], i+1)
                
            # Add GeoTIFF tags for better compatibility
            dst.update_tags(
                TIFFTAG_RESOLUTIONUNIT="2",  # Pixels per inch
                TIFFTAG_XRESOLUTION='300',
                TIFFTAG_YRESOLUTION='300'
            )
            
        print(f"Successfully saved GeoTIFF to {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save GeoTIFF: {str(e)}")
        
    # Verify the output
    try:
        with rasterio.open(output_path) as src:
            actual_bounds = src.bounds
            print(f"Verified bounds: {actual_bounds}")
            print(f"Original bounds: {(lon_min, lat_min, lon_max, lat_max)}")
    except Exception as e:
        print(f"Warning: Could not verify output file: {str(e)}")

def save_tif_gdal(image, output_path, lat, lon):
    """
    Save image data as a georeferenced TIFF file using GDAL.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D or 3D image array (bands, height, width)
    output_path : str
        Path where the GeoTIFF will be saved
    lat : numpy.ndarray
        2D array of latitude values
    lon : numpy.ndarray
        2D array of longitude values
    """
    from osgeo import gdal, osr
    import numpy as np
    from pyproj import Proj, transform
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import rasterio
    
    # Input validation
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim != 3:
        raise ValueError("Image must be 2D or 3D array")
    
    channels, rows, cols = image.shape
    
    # Ensure lat/lon are 2D arrays
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("Latitude and longitude arrays must be 2D")
    
    # Define the projection transformation (WGS84 to UTM Zone 33N as an example)
    wgs84 = Proj(proj="latlong", datum="WGS84")
    utm = Proj(proj="utm", zone=16, datum="WGS84")
    
    # Convert latitude and longitude to projected UTM coordinates
    x, y = transform(wgs84, utm, lon, lat)
    
    # Get the exact bounds in UTM
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    
    # Calculate transform using projected bounds
    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)
    
    # Define the CRS for UTM Zone 33N
    crs = CRS.from_proj4(utm.srs)
    
    # Metadata for the GeoTIFF
    metadata = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': channels,
        'dtype': image.dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',  # Add compression
        'tiled': True,      # Enable tiling for better performance
        'interleave': 'band'
    }
    
    try:
        with rasterio.open(output_path, 'w', **metadata) as dst:
            # Write the data
            for i in range(channels):
                dst.write(image[i, :, :], i+1)
                
            # Add GeoTIFF tags for better compatibility
            dst.update_tags(
                TIFFTAG_RESOLUTIONUNIT="2",  # Pixels per inch
                TIFFTAG_XRESOLUTION='300',
                TIFFTAG_YRESOLUTION='300'
            )
            
        print(f"Successfully saved GeoTIFF to {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save GeoTIFF: {str(e)}")
        
    # Verify the output
    try:
        with rasterio.open(output_path) as src:
            actual_bounds = src.bounds
            print(f"Verified bounds: {actual_bounds}")
            print(f"Original bounds (projected): {(x_min, y_min, x_max, y_max)}")
    except Exception as e:
        print(f"Warning: Could not verify output file: {str(e)}")

def verify_georeference(output_path, original_lat, original_lon):
    """
    验证生成的GeoTIFF文件的地理参考是否正确
    
    Parameters:
    -----------
    output_path : str
        GeoTIFF文件路径
    original_lat : numpy.ndarray
        原始纬度数组
    original_lon : numpy.ndarray
        原始经度数组
    """
    from osgeo import gdal
    import numpy as np
    
    dataset = gdal.Open(output_path)
    if dataset is None:
        raise RuntimeError("Could not open the output file")
    
    geotransform = dataset.GetGeoTransform()
    
    # 计算四个角点坐标
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    # GDAL坐标转换函数
    def pixel_to_coord(x, y, geotransform):
        lon = geotransform[0] + x * geotransform[1] + y * geotransform[2]
        lat = geotransform[3] + x * geotransform[4] + y * geotransform[5]
        return lon, lat
    
    corners = {
        'upper_left': pixel_to_coord(0, 0, geotransform),
        'upper_right': pixel_to_coord(width-1, 0, geotransform),
        'lower_left': pixel_to_coord(0, height-1, geotransform),
        'lower_right': pixel_to_coord(width-1, height-1, geotransform)
    }
    
    print("\nCorner Coordinates Comparison:")
    print("Original corners:")
    print(f"Upper left: ({original_lon[0,0]}, {original_lat[0,0]})")
    print(f"Upper right: ({original_lon[0,-1]}, {original_lat[0,-1]})")
    print(f"Lower left: ({original_lon[-1,0]}, {original_lat[-1,0]})")
    print(f"Lower right: ({original_lon[-1,-1]}, {original_lat[-1,-1]})")
    
    print("\nGeoTIFF corners:")
    for corner, coords in corners.items():
        print(f"{corner}: {coords}")
    
    # 计算偏差
    original_corners = {
        'upper_left': (original_lon[0,0], original_lat[0,0]),
        'upper_right': (original_lon[0,-1], original_lat[0,-1]),
        'lower_left': (original_lon[-1,0], original_lat[-1,0]),
        'lower_right': (original_lon[-1,-1], original_lat[-1,-1])
    }
    
    print("\nOffset analysis:")
    for corner in corners.keys():
        lon_diff = corners[corner][0] - original_corners[corner][0]
        lat_diff = corners[corner][1] - original_corners[corner][1]
        print(f"{corner} offset:")
        print(f"Longitude difference: {lon_diff:.6f} degrees")
        print(f"Latitude difference: {lat_diff:.6f} degrees")
    
    dataset = None
    
if __name__=='__main__':
    input_path = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeLanier\2024October7\SatelliteImages\Sentinel2\L2_acolite\S2B_MSIL2_20241007T161049_N0511_R140_T16SGC_20241007T195344.SAFE\S2B_MSI_2024_10_07_16_24_06_T16SGC_L2R.nc'
    output_path = os.path.basename(input_path).split('.')[0]+'.tif'
    
    ds = h5py.File(input_path,'r')
    
    if os.path.basename(input_path)[0:3]=='S2A':
        G = np.array(ds['rhos_560'])
        SWIR = np.array(ds['rhos_1614'])
        
    elif os.path.basename(input_path)[0:3]=='S2B':
        G = np.array(ds['rhos_559'])
        SWIR = np.array(ds['rhos_1610'])
    
    elif os.path.basename(input_path)[0]=='L':
        G = np.array(ds['rhos_561'])
        SWIR = np.array(ds['rhos_1608'])
        
    else:
        print('The format of input image cannot be recognized!')
    
    lat = np.array(ds['lat'])
    lon = np.array(ds['lon'])
    
    water_mask = NDWI(G, SWIR)
    
    # water_mask = Speckle_removal(water_mask)
    
    save_tif_gdal(water_mask, output_path, lat, lon)
    
    output_shp_path = os.path.basename(input_path).split('.')[0]+'.shp'
    tif2shp(output_path, output_shp_path)
