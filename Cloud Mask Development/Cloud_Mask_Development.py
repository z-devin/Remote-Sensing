#!wget https://storage.googleapis.com/krishna-skytruth-dev/ece471/homework_1/s2_santafe.zip
#!unzip -oq s2_santafe.zip

#!ls s2_santafe

#!pip install rasterio
#!pip install gdal

from osgeo import gdal, osr, gdalconst
import rasterio
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from shapely.geometry import box, mapping
import json
import shutil
import glob

"""#**TASK 1: VISUAL INSPECTION**"""

raster_file = 's2_santafe/sentinel-2:L1C_2018-04-04.tif'
src = rasterio.open(raster_file)

"""Metadata"""

src.profile

"""The data is stored as uint16, which means we will need to convert it to uint8 or normalize it to [0,1] before displaying it.

Spatial Bounds
"""

src.bounds

"""Coordinate Reference System"""

src.crs

"""Pixel to Geographic Conversion"""

src.transform

"""Store the Data as an Array"""

arr = src.read()

arr.shape

"""This makes sense as there are 7 bands. The 1504x1469 represent the spatial resolution in pixels.

Pre-scaling Visual Inspection
"""

red = arr[0, :, :]/65535
green = arr[1, :, :]/65535
blue = arr[2, :, :]/65535
nir = arr[3, :, :]/65535
swir1 = arr[4, :, :]/65535
swir2 = arr[5, :, :]/65535
alpha = arr[6, :, :]/65535

"""The uint16 data type is 16 bits long and can have values from 0 to 65535. To normalize the values to the range [0,1], we divide each value by 65535.

RGB Visualization
"""

rgb = np.stack([red, green, blue], axis=-1)
plt.figure(dpi=180)
plt.imshow(rgb)
plt.axis('off')
plt.show()

"""NRG Visualization"""

nrg = np.stack([nir, red, green], axis=-1)
plt.figure(dpi=180)
plt.imshow(nrg)
plt.axis('off')
plt.show()

"""The images seems very dark, however we can see little bits of white at the bottom left corner.
Let's make a function that can create a visualization for any single or three-band combos and performs a min-max stretch with a cumulative count cut.
"""

def min_max_stretch(array, min_percentile=5, max_percentile=95, output_range=(0, 1)):
    valid_values = array[array > 0]

    if valid_values.size == 0:
        return array

    min_value = np.percentile(valid_values, min_percentile)
    max_value = np.percentile(valid_values, max_percentile)

    stretched = np.clip(array, min_value, max_value)

    out_min, out_max = output_range
    stretched = (stretched - min_value) / (max_value - min_value)
    stretched = stretched * (out_max - out_min) + out_min

    stretched[array == 0] = 0

    return stretched

def visualize_raster(image_path, bands=(1, 2, 3), min_percentile=5, max_percentile=95):

    with rasterio.open(image_path) as src:
        arr = src.read()

    if len(bands) == 1:
        band = arr[bands[0] - 1, :, :]
        stretched_band = min_max_stretch(band, min_percentile, max_percentile)

        plt.figure(dpi=300)
        plt.imshow(stretched_band)

    elif len(bands) == 3:
        selected_bands = [arr[b - 1, :, :] for b in bands]
        stretched_bands = [min_max_stretch(band, min_percentile, max_percentile) for band in selected_bands]

        rgb_image = np.dstack(stretched_bands)
        plt.figure(dpi=300)
        plt.imshow(rgb_image)
    else:
        raise ValueError("Only single or three-band combinations are allowed.")

    plt.axis('off')
    plt.show()

"""The min_max_stretch function clips the dataset to the range [min_value, max_value] (by default, 5th percentile and 95th percentile). Any pixel values below min_value are set to min_value, and any values above max_value are set to max_value. This is to remove the extreme outliers which would ruin the stretch. It then normalizes the image's pixel values to the range [0,1].

The visualize_raster function uses the min_max_stretch function and applies it to the chosen bands and returns the a visual of the processed image.
"""

visualize_raster(raster_file, bands=(1, 2, 3))

"""Some of the pictures have clouds which will definitely become an issue when workng with this data. The clouds also cast a shadow onto the ground below it, which will also need to be dealt with.

Some pictures are also cut in half. We will deal with them in task 4.
"""

visualize_raster('s2_santafe/sentinel-2:L1C_2018-04-07.tif', bands=(1, 2, 3))

"""Some images are also cut in half or only show the left half.

#**TASK 2: DATA PROCESSING**

Setting up definitions.

CRS Justification:
I decided on using EPSG:32613 as the target CRS because UTM gives good accuracy/minimal distortion in local/small regions as it divides the earth into 60 zones. UTM also uses meters which is easy to work with.

Resolution Justification:
I decided to use a resolution of 10m because although it takes up more memory than 20m or 60m, vegetation analysis, NDVI, relies on the red (10m) and NIR (10m) bands so having a higher resolution for it to be detected accurately.
"""

input_folder = 's2_santafe'
output_folder = 's2_santafe_processed'
epsg = 32613
resolution = 10
os.makedirs(output_folder, exist_ok=True)

"""Setting up a function that helps us convert the crs and resolution of all images to our desired values. Also optionally clips the images to a defined bounding box.

For the resampling algorithm, I chose GRA_NearestNeighbor because it is the least computationally heavy and preserves the original pixel values without interpolation, which is important to get accurate values in our context.
"""

def reproject_and_resample(input_path, output_path, dst_srs_wkt, xres, yres,
                           output_bounds=None, resampleAlg=gdalconst.GRA_NearestNeighbour):

    input_directory = "s2_santafe"
    safe_input_path = os.path.join(input_directory, os.path.basename(input_path))

    ds_in = gdal.Open(safe_input_path)

    warp_opts = gdal.WarpOptions(
        format='GTiff',
        dstSRS=dst_srs_wkt,
        xRes=xres,
        yRes=yres,
        resampleAlg=resampleAlg,
        outputBounds=output_bounds,
        targetAlignedPixels=True
    )

    result = gdal.Warp(destNameOrDestDS=output_path, srcDSOrSrcDSTab=ds_in, options=warp_opts)
    result.FlushCache()
    del result

"""This function takes an image and returns the bounding box of the image.

This data will be compared to find the smallest bounding box across all images in the set.
"""

def get_raster_bbox(dataset):
    gt = dataset.GetGeoTransform()
    w = dataset.RasterXSize
    h = dataset.RasterYSize

    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + w * gt[1]
    min_y = max_y + h * gt[5]
    return (min_x, min_y, max_x, max_y)

"""This function takes an image and clips it to the bounds it is given."""

def clip_raster(input_path, output_path, output_bounds):
    ds_in = gdal.Open(input_path)

    warp_opts = gdal.WarpOptions(
        format='GTiff',
        outputBounds=output_bounds,
        cutlineBlend=0,
        cropToCutline=True,
    )

    result = gdal.Warp(output_path, ds_in, options=warp_opts)
    if result is None:
        raise RuntimeError(f"GDAL Warp (clipping) failed for {input_path}")
    else:
        result.FlushCache()

"""This function takes an input folder and uses the previous functions to:
- Warp each tif file to the target CRS (EPSG:32613) and target resolution (10m).
- Then computes the smallest/common bounding box amongst the warped rasters.
- Then clips each raster to the computed bounding box.
- Takes the now processed rasters and converts them to a 3D numpy stack
- Finally, creates a geojson representation of the clipped spatial extent.
"""

def process_images(input_folder, output_folder, target_epsg, final_resolution):
    os.makedirs(output_folder, exist_ok=True)
    common_crs = f"EPSG:{target_epsg}"

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

    srs_32613 = osr.SpatialReference()
    srs_32613.ImportFromEPSG(target_epsg)
    dst_srs_wkt = srs_32613.ExportToWkt()

    # Warping
    warped_paths = []
    warped_paths = []
    for img_path in images:
        base = os.path.basename(img_path)
        base_no_ext, ext = os.path.splitext(base)
        warped_filename = f"{base_no_ext}_warped{ext}"
        warped_path = os.path.join(output_folder, warped_filename)

        reproject_and_resample(
            input_path=os.path.join(input_folder, img_path),
            output_path=warped_path,
            dst_srs_wkt=dst_srs_wkt,
            xres=final_resolution,
            yres=final_resolution
        )
        warped_paths.append(warped_path)

    print("Warping Complete.")

    # Compute intersection
    all_boxes = []
    for wpath in warped_paths:
        ds = gdal.Open(wpath)
        bbox = get_raster_bbox(ds)
        all_boxes.append(bbox)

    min_x = max(b[0] for b in all_boxes)
    min_y = max(b[1] for b in all_boxes)
    max_x = min(b[2] for b in all_boxes)
    max_y = min(b[3] for b in all_boxes)

    final_extent = box(min_x, min_y, max_x, max_y)
    print("Final Intersection (EPSG:32613):", final_extent.bounds)

    # Clip each warped file
    clipped_paths = []
    for wpath in warped_paths:
        base_warped = os.path.basename(wpath)
        base_no_ext, ext = os.path.splitext(base_warped)
        final_name = base_no_ext.replace("_warped", "") + ext
        clipped_path = os.path.join(output_folder, final_name)

        clip_raster(
            input_path=wpath,
            output_path=clipped_path,
            output_bounds=(min_x, min_y, max_x, max_y)
        )
        clipped_paths.append(clipped_path)
    print("Clipping Complete.")

    # Stack array
    raster_stack = []
    for cpath in clipped_paths:
        ds = gdal.Open(cpath)
        arr = ds.ReadAsArray()  # shape: (bands, rows, cols)
        raster_stack.append(arr)
    raster_stack = np.array(raster_stack)  # (scenes, bands, rows, cols)
    print("Final raster stack shape:", raster_stack.shape)

    # Write GeoJSON
    geojson_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(final_extent),
                "properties": {
                    "crs": common_crs,
                    "resolution": final_resolution,
                    "width": raster_stack.shape[-1],
                    "height": raster_stack.shape[-2],
                }
            }
        ]
    }
    geojson_path = os.path.join(output_folder, "final_extent.geojson")
    with open(geojson_path, "w") as f:
        json.dump(geojson_dict, f, indent=2)

    print("GeoJSON saved to:", geojson_path)

    # Remove intermediate files
    print("Removing intermediate files.")
    for wpath in warped_paths:
        os.remove(wpath)

    return raster_stack, clipped_paths, geojson_path


stack, final_files, geojson = process_images(input_folder, output_folder, epsg, resolution)

#!ls s2_santafe_processed/

"""Confirmation that the, size in pixels, Bounding box, and CRS, are all consistent throughout the dataset."""

def load_geojson_metadata(geojson_path):
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    feature = geojson_data["features"][0]
    bbox = feature["geometry"]["coordinates"][0]
    crs = feature["properties"]["crs"]
    resolution = feature["properties"]["resolution"]

    min_x = min(pt[0] for pt in bbox)
    max_x = max(pt[0] for pt in bbox)
    min_y = min(pt[1] for pt in bbox)
    max_y = max(pt[1] for pt in bbox)

    return (min_x, min_y, max_x, max_y), crs, resolution

def get_raster_metadata(tif_path):
    ds = gdal.Open(tif_path)

    width = ds.RasterXSize
    height = ds.RasterYSize

    gt = ds.GetGeoTransform()
    min_x = gt[0]
    max_y = gt[3]
    max_x = min_x + width * gt[1]
    min_y = max_y + height * gt[5]

    crs_wkt = ds.GetProjection()
    crs = osr.SpatialReference()
    crs.ImportFromWkt(crs_wkt)
    crs_epsg = crs.GetAttrValue("AUTHORITY", 1)

    x_res = abs(gt[1])
    y_res = abs(gt[5])
    return (width, height), (min_x, min_y, max_x, max_y), crs_epsg, x_res, y_res

def check_tif_consistency(directory, geojson_path):

    expected_bbox, expected_crs, expected_res = load_geojson_metadata(geojson_path)
    print(f"Expected CRS: {expected_crs}")
    print(f"Expected Bounding Box: {expected_bbox}")
    print(f"Expected Resolution: {expected_res}m")

    tif_files = [f for f in os.listdir(directory) if f.lower().endswith(".tif")]

    for tif in tif_files:
        tif_path = os.path.join(directory, tif)
        metadata = get_raster_metadata(tif_path)

        size, bbox, crs, x_res, y_res = metadata

        if crs != expected_crs.split(":")[1]:
            print(f"[Mismatch] {tif} has a different CRS: {crs} (expected {expected_crs})")

        if bbox != expected_bbox:
            print(f"[Mismatch] {tif} has a different bounding box: {bbox}")

        if x_res != expected_res or y_res != expected_res:
            print(f"[Mismatch] {tif} has a different resolution: {x_res}m, {y_res}m")

    print("Consistency check complete.")

geojson_path = "s2_santafe_processed/final_extent.geojson"
check_tif_consistency("s2_santafe_processed", geojson_path)

visualize_raster('s2_santafe_processed/sentinel-2:L1C_2018-04-04.tif', bands=(1, 2, 3))

"""The image comes out well post processing.

#**TASK 3: REFLECTANCE VALUES**
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import glob

def compute_histogram_efficient(input_directory, bins=200, skip_band=7):
    tif_files = glob.glob(os.path.join(input_directory, "*.tif"))

    all_band_histograms = {}
    bin_edges = None

    for tif_file in tif_files:
        dataset = gdal.Open(tif_file)
        if dataset is None:
            print(f"Skipping {tif_file}, could not open.")
            continue

        num_bands = dataset.RasterCount

        for band_idx in range(1, num_bands + 1):
            if band_idx == skip_band:
                continue

            band = dataset.GetRasterBand(band_idx)
            band_data = band.ReadAsArray()

            if band_data is None:
                print(f"Skipping empty band {band_idx} in {tif_file}")
                continue

            band_data = band_data.astype(np.float32).flatten()

            nodata_value = band.GetNoDataValue()
            if nodata_value is not None:
                band_data = band_data[band_data != nodata_value]

            hist, edges = np.histogram(band_data, bins=bins, range=(band_data.min(), band_data.max()))

            if bin_edges is None:
                bin_edges = edges

            if band_idx not in all_band_histograms:
                all_band_histograms[band_idx] = np.zeros_like(hist, dtype=np.int64)
            all_band_histograms[band_idx] += hist

    plt.figure(figsize=(12, 6))

    for band_idx, hist_values in all_band_histograms.items():
        plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), alpha=0.6, label=f'Band {band_idx}')

    plt.xlabel("Reflectance Value")
    plt.ylabel("Frequency")
    plt.title("Reflectance Value Distribution Across Entire Dataset (Skipping Band 7)")
    plt.legend()
    plt.grid(True)
    plt.show()

compute_histogram_efficient('s2_santafe_processed')

"""The frequencies are a lot higher for reflectance values at on the lower end, meaning most pixels have low reflectance.

Each band definitely has their own respective peaks and spread. However, the bands overlap significantly. This could definitely pose a challenge when trying to differentiate between the bands.

The spike of zero reflectance values most likely comes from the cut off images, we will deal with those in the next task.

#**TASK 4: DATA FILTERING/CLEANING**

Omitting pictures that have less than 90% of valid pixels (to get rid of the images that are cut in half).

We do this using the alpha band, which tells us about the validity of each pixel. If the percentage is below 90, we omit the image.

The function also saves the filtered images to a new directory named 's2_santafe_processed_filtered' while leaving the processed dataset untouched.
"""

def filter_out_invalid_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

    for image in images:
        image_path = os.path.join(input_folder, image)
        ds = gdal.Open(image_path)
        alpha_band = ds.GetRasterBand(7).ReadAsArray()
        percent_valid = np.mean(alpha_band == 65535) * 100

        if percent_valid >= 90:
            dst_path = os.path.join(output_folder, image)
            shutil.copy2(image_path, dst_path)

    print("Filtering complete.")

filter_out_invalid_images('s2_santafe_processed', 's2_santafe_processed_filtered')

#!ls s2_santafe_processed_filtered

compute_histogram_efficient('s2_santafe_processed_filtered')

"""The frequency of reflectance values that were zeros decrease significantly, meaning a lot of them came from the cut off images.

#**TASK 5: CLOUD MASK**

To filter out the clouds, we first need to create a function to determine whether the pixels on an image are clouds or not.

The function below takes a probability based approach where different weighted spectral indices contribute to whether or not a pixel is a cloud or not. If the final probability exceeeds 50%, then we classify it as a cloud.

I ended up using the spectral indices:
ndci, ndsi, cpi, mbcs, and brightness. The purposes of each are commented in the function.
"""

from scipy.ndimage import binary_dilation

def detect_clouds(image_path):
    ds = gdal.Open(image_path)

    red = ds.GetRasterBand(1).ReadAsArray().astype(np.float32) / 65535
    green = ds.GetRasterBand(2).ReadAsArray().astype(np.float32) / 65535
    blue = ds.GetRasterBand(3).ReadAsArray().astype(np.float32) / 65535
    nir = ds.GetRasterBand(4).ReadAsArray().astype(np.float32) / 65535
    swir1 = ds.GetRasterBand(5).ReadAsArray().astype(np.float32) / 65535
    swir2 = ds.GetRasterBand(6).ReadAsArray().astype(np.float32) / 65535

    ds = None

    ndci = (blue - swir1) / (blue + swir1 + 1e-6)  # NDCI: Strong for general cloud detection
    ndsi = (green - swir1) / (green + swir1 + 1e-6)  # NDSI: Helps separate clouds from snow
    cpi = (nir + swir1 + swir2 + blue + green + red) / 6  # CPI: General cloud probability indicator
    mbcs = 1.0 - (swir2 / (blue + green + red + 1e-6))  # MBCS: Captures thick clouds well
    brightness= ((blue + green + red) / 3)  # Brightness: Detects very bright clouds
    shadow = (swir2 + nir + brightness) / 3  # Shadow: Detects shadows

    # Thresholds
    ndci_mask = ndci > -0.08
    ndsi_mask = ndsi > 0.4

    cpi_mask = cpi > 0.05
    mbcs_mask = mbcs > 0.7
    brightness_mask = brightness > 0.035
    shadow_mask = shadow < 0.013

    # Weights
    ndci_weight = 0.30
    ndsi_weight = 0.15
    cpi_weight = 0.16
    mbcs_weight = 0.15
    brightness_weight = 0.24
    shadow_weight = 0.4   # I include the shadow with high weight.

    # Compute probabilities
    cloud_probability = (
        ndci_weight * (ndci > -0.08) +
        ndsi_weight * (ndsi > -0.1) +
        cpi_weight * (cpi > 0.05) +
        mbcs_weight * (mbcs > 0.8) +
        brightness_weight * (brightness > 0.035) +
        shadow_weight * (shadow < 0.2)
    )

    final_cloud_mask = cloud_probability > 0.5  # Cloud if probability > 50%

    # Ignore this bit, I was testing a different system to aggregate the different indexes.
    # Voting system
    # cloud_votes = ndci_mask.astype(int) + ndsi_mask.astype(int) + cpi_mask.astype(int) + mbcs_mask.astype(int) + brightness_mask.astype(int) + ci_mask.astype(int) + ci2_mask.astype(int)
    # final_cloud_mask = cloud_votes >= 4


    # Plotting
    # fig, axes = plt.subplots(6, 2, figsize=(12, 35))

    # def plot_with_colorbar(ax, data, title, cmap='viridis'):
    #     img = ax.imshow(data, cmap=cmap)
    #     ax.set_title(title)
    #     cbar = plt.colorbar(img, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    #     cbar.ax.set_ylabel('Index Value')

    # plot_with_colorbar(axes[0, 0], ndci, "NDCI (Cloud Index)", cmap='coolwarm')
    # axes[0, 1].imshow(ndci_mask, cmap='gray')
    # axes[0, 1].set_title("NDCI Cloud Mask")
    # plot_with_colorbar(axes[1, 0], ndsi, "NDSI (Snow Separation)", cmap='coolwarm')
    # axes[1, 1].imshow(ndsi_mask, cmap='gray')
    # axes[1, 1].set_title("NDSI Cloud Mask")
    # plot_with_colorbar(axes[2, 0], cpi, "CPI (Cloud Probability Index)", cmap='plasma')
    # axes[2, 1].imshow(cpi_mask, cmap='gray')
    # axes[2, 1].set_title("CPI Cloud Mask")
    # plot_with_colorbar(axes[3, 0], mbcs, "MBCS (Multi-Band Cloud Score)", cmap='viridis')
    # axes[3, 1].imshow(mbcs_mask, cmap='gray')
    # axes[3, 1].set_title("MBCS Cloud Mask")
    # plot_with_colorbar(axes[4, 0], brightness, "Brightness", cmap='viridis')
    # axes[4, 1].imshow(brightness_mask, cmap='gray')
    # axes[4, 1].set_title("Brightness")
    # plot_with_colorbar(axes[5, 0], shadow_mask, "Shadow Mask", cmap='viridis')
    # axes[5, 1].imshow(shadow_mask, cmap='gray')
    # axes[5, 1].set_title("Shadow Mask")

    # plt.tight_layout()

    return final_cloud_mask

cloud_mask = detect_clouds('s2_santafe_processed_filtered/sentinel-2:L1C_2018-06-13.tif')

plt.figure(figsize=(8, 8))
plt.imshow(cloud_mask, cmap='gray')
plt.title("Cloud Mask")
plt.show()

"""The function seems to perform pretty well at classifying the clouds. It seems to perform well for clearly defined clouds, but seems to fail when the clouds are thin. It also tends to pick up pixels that are very white. If I had more time, I could optimize the weights or find other useful indexes that may be useful.

Now we apply this to all images in our processed dataset and output the masked images to a new directory named 's2_santafe_processed_filtered_masked'.
"""

def apply_cloud_mask_to_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))

    for tif_file in tif_files:
        cloud_mask = detect_clouds(tif_file)
        src_ds = gdal.Open(tif_file, gdal.GA_ReadOnly)

        out_file = os.path.join(output_dir, os.path.basename(tif_file))

        driver = gdal.GetDriverByName("GTiff")
        band_type = src_ds.GetRasterBand(1).DataType
        out_ds = driver.Create(
            out_file,
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            src_ds.RasterCount,
            band_type
        )

        out_ds.SetGeoTransform(src_ds.GetGeoTransform())
        out_ds.SetProjection(src_ds.GetProjection())

        # Band by band
        for band_idx in range(1, src_ds.RasterCount + 1):
            in_band = src_ds.GetRasterBand(band_idx)
            data = in_band.ReadAsArray()
            data[cloud_mask == 1] = 0

            out_band = out_ds.GetRasterBand(band_idx)
            out_band.WriteArray(data)

        out_ds.FlushCache()
        out_ds = None
        src_ds = None

    print("Cloud-mask Complete.")

apply_cloud_mask_to_directory('s2_santafe_processed_filtered', 's2_santafe_processed_filtered_masked')

visualize_raster('s2_santafe_processed_filtered_masked/sentinel-2:L1C_2018-06-13.tif', bands=(1, 2, 3))

"""The cloud mask seems to work decently, it seems to struggle for the edges of the clouds and other very dark pixels. These can definitely be improved with more time.

#**TASK 6: VEGETATION, SNOW, CLOUDS, BRIGHTNESS**

This function analyzes a stack of Sentinel-2 images and finds the best scene for each of the following categories:
- Most Vegetation: Scene with the highest NDVI.
- Most Snow: Scene with the highest NDSI.
- Most Clouds: Scene with the highest cloud coverage.
- Highest Brightness: Scene with the highest mean brightness.

This function reads the spectral bands from each image and computes the mean value for each index across the entire image for the vegatation, snow, and brightness. The cloud coverage was calculated using the detect_clouds function and computes the percentage of cloud pixels in each image, then finds the max.
"""

def compute_scene_indices(directory):
    scenes = glob.glob(os.path.join(directory, "*.tif"))

    best_scenes = {
        "most_vegetation": None,
        "most_snow": None,
        "most_clouds": None,
        "highest_brightness": None
    }

    max_values = {
        "most_vegetation": -np.inf,
        "most_snow": -np.inf,
        "most_clouds": -np.inf,
        "highest_brightness": -np.inf
    }

    for scene in scenes:
        with rasterio.open(scene) as src:
            red = src.read(1).astype(np.float32) / 65535
            green = src.read(2).astype(np.float32) / 65535
            blue = src.read(3).astype(np.float32) / 65535
            nir = src.read(4).astype(np.float32) / 65535
            swir1 = src.read(5).astype(np.float32) / 65535

            ndvi = (nir - red) / (nir + red + 1e-6)  # NDVI for vegetation
            ndsi = (green - swir1) / (green + swir1 + 1e-6)  # NDSI for snow
            brightness = (red + green + blue) / 3  # Brightness

            cloud_mask = detect_clouds(scene)
            cloud_percentage = np.sum(cloud_mask == 1) / cloud_mask.size

            ndvi_mean = np.nanmean(ndvi)
            ndsi_mean = np.nanmean(ndsi)
            brightness_mean = np.nanmean(brightness)

            if ndvi_mean > max_values["most_vegetation"]:
                max_values["most_vegetation"] = ndvi_mean
                best_scenes["most_vegetation"] = scene

            if ndsi_mean > max_values["most_snow"]:
                max_values["most_snow"] = ndsi_mean
                best_scenes["most_snow"] = scene

            if cloud_percentage > max_values["most_clouds"]:
                max_values["most_clouds"] = cloud_percentage
                best_scenes["most_clouds"] = scene

            if brightness_mean > max_values["highest_brightness"]:
                max_values["highest_brightness"] = brightness_mean
                best_scenes["highest_brightness"] = scene

    return best_scenes

best_scenes = compute_scene_indices('s2_santafe_processed_filtered_masked')
print("Best Scenes:")
for category, scene in best_scenes.items():
    file_name = os.path.basename(scene)
    print(f"{category}: {file_name}")

"""The scene with the most vegetation, snow, and brightness produced by my algorithm matches up with visual inspection. However, unfortunately, the cloud scene does not, it seems to pick the scene with the most snow as it mistakes the bright snow as clouds. This is mainly due to an imperfect cloud mask, which can be improved with more time.

#**TASK 7: COMPOSITE IMAGES**

We first define a function similar to visualize_raster from before but for arrays. This is to print the composite images later.
"""

def visualize_array(image_array, bands=(1, 2, 3), min_percentile=5, max_percentile=95):

    if len(bands) == 1:
        band = image_array[bands[0] - 1, :, :]
        stretched_band = min_max_stretch(band, min_percentile, max_percentile)

        plt.figure(dpi=300)
        plt.imshow(stretched_band, cmap='gray')

    elif len(bands) == 3:
        selected_bands = [image_array[b - 1, :, :] for b in bands]
        stretched_bands = [min_max_stretch(band, min_percentile, max_percentile) for band in selected_bands]

        rgb_image = np.dstack(stretched_bands)
        plt.figure(dpi=300)
        plt.imshow(rgb_image)
    else:
        raise ValueError("Only single or three-band combinations are allowed.")

    plt.axis('off')
    plt.show()

"""This function computes the composite images from a temporal stack of rasters. It reads the raster data band by band and calculates each composite pixel-wise across all images, ignoring the zero values from the cloud mask.

To compute:
- Mean Composite: It takes the average value per pixel.
- Median Composite: It takes the middle value per pixel.
- Min Composite: It takes the minimum value per pixel.
- Max Composite: It takes the max value per pixel.
- Greenest Composite: It takes the pixel with the highest NDVI value.
"""

def compute_composites(directory):
    images = sorted(glob.glob(os.path.join(directory, "*.tif")))

    dataset = gdal.Open(images[0], gdal.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    num_bands = dataset.RasterCount

    sum_array = np.zeros((num_bands, height, width), dtype=np.float64)
    min_array = np.full((num_bands, height, width), np.inf, dtype=np.float64)
    max_array = np.full((num_bands, height, width), -np.inf, dtype=np.float64)
    count = np.zeros((num_bands, height, width), dtype=np.int32)

    greenest_array = np.zeros((num_bands, height, width), dtype=np.float32)
    max_ndvi = np.full((height, width), -np.inf, dtype=np.float32)

    all_data = []

    for img_path in images:
        dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
        data = np.zeros((num_bands, height, width), dtype=np.float32)

        for band_idx in range(num_bands):
            band = dataset.GetRasterBand(band_idx + 1)
            data[band_idx, :, :] = band.ReadAsArray().astype(np.float32) / 65535

        nir = data[3, :, :]
        red = data[0, :, :]
        ndvi = (nir - red) / (nir + red + 1e-6)

        valid_mask = np.all(data > 0, axis=0)

        update_mask = valid_mask & (ndvi > max_ndvi)
        max_ndvi[update_mask] = ndvi[update_mask]
        greenest_array[:, update_mask] = data[:, update_mask]

        all_data.append(np.where(valid_mask, data, np.nan))

        sum_array += np.where(valid_mask, data, 0)
        min_array = np.where(valid_mask, np.minimum(min_array, data), min_array)
        max_array = np.where(valid_mask, np.maximum(max_array, data), max_array)
        count += valid_mask.astype(np.int32)

        dataset = None

    all_data = np.array(all_data)

    mean_array = np.divide(sum_array, count, where=count > 0)
    median_array = np.nanmedian(all_data, axis=0)

    composites = {
        "mean": mean_array,
        "median": median_array,
        "min": min_array,
        "max": max_array,
        "greenest": greenest_array
    }

    return composites

"""First, lets run the function on the unmasked set of images."""

composites = compute_composites('s2_santafe_processed_filtered')
print("Composites Computed.")
for name, composite in composites.items():
    print(f"{name}")
    visualize_array(composite)

"""Now, the masked set of images."""

composites = compute_composites('s2_santafe_processed_filtered_masked')
print("Composites Computed.")
for name, composite in composites.items():
    print(f"{name}")
    visualize_array(composite)

"""Implementing the cloud mask definitely helped as it doesn't take into account the majority of the very bright cloud pixels and misinformation.

The implementation of the cloud mask made the most impact on the max, min, and greenest composite.

- The unmasked max composite shows a lot of clouds due to the clouds pixels being a lot brighter than the others, while the masked max composite looks a bit messy most likely due to an imperfect cloud mask. However, the other composites look good.

- The unmasked min composite shows a lot of dark cloud shadows that implementing the cloud mask definitely helped get rid of. However, although the masked min composite got rid of the majority of them, you can still see the edges of a few cloud shadows that the mask didn't get rid of in the final composite image.

- For the greenest composite, the mask definitely helped get rid of a lot of the little traces of clouds.

A better cloud mask would definitely improve the overall quality of the composites."""