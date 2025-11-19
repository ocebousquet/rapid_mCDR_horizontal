import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import PyCO2SYS as pyco2
import cartopy.crs as ccrs
from netCDF4 import Dataset
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec


def flat(fld, **kwargs):
    """
    Converts MITgcm MDS data into a global 2D field.

    This function handles data arrays with **2 to 5 dimensions** by calling `flat2D`
    for each 2D slice in the higher-dimensional array.

    Parameters:
        - fld (np.ndarray): Input data array (must have between 2 and 5 dimensions).
        - **kwargs: Additional arguments passed to `flat2D`.

    Returns:
        - np.ndarray: Flattened global field.

    Raises:
        - SystemExit: If the input has an unsupported number of dimensions.
    """

    ndims = len(fld.shape)  # Number of dimensions in the input array

    if ndims == 2:
        # Directly flatten 2D data
        gfld = flat2D(fld, **kwargs)
    elif ndims == 3:
        # Iterate over the first axis (e.g., depth levels)
        gfld = [flat2D(fld[a, :, :], **kwargs) for a in range(fld.shape[0])]
    elif ndims == 4:
        # Iterate over two axes (e.g., time and depth)
        gfld = [[flat2D(fld[a, b, :, :], **kwargs) for b in range(fld.shape[1])] for a in range(fld.shape[0])]
    elif ndims == 5:
        # Iterate over three axes (e.g., time, depth, and another dimension)
        gfld = [[[flat2D(fld[a, b, c, :, :], **kwargs) for c in range(fld.shape[2])] for b in range(fld.shape[1])] for a
                in range(fld.shape[0])]
    else:
        print("Error: Unsupported number of dimensions.")
        print("Only 2D to 5D arrays are allowed.")
        sys.exit(__doc__)  # Exits the program with the function's docstring as an error message

    return np.array(gfld)  # Convert the output list to a NumPy array


def flat2D(fld, center='Atlantic'):
    """
    Converts a 2D MITgcm MDS field into a global projection.

    This function reconstructs the field by rearranging the data
    into a global 2D grid considering different hemispheres and
    special handling for the Arctic region.

    Parameters:
        - fld (np.ndarray): 2D field to be transformed.
        - center (str, optional): Centering of the output grid.
          Options:
            - 'Atlantic' (default): Positions Atlantic in the center.
            - 'Pacific': Positions Pacific in the center.

    Returns:
        - np.ndarray: The flattened global 2D field.
    """

    nx = fld.shape[1]  # Number of longitude points
    ny = fld.shape[0]  # Number of latitude points

    # Ensure 'n' is always at least 1 (prevents division errors)
    n = max(1, ny // nx // 4)

    # Split the data into eastern and western hemispheres
    eastern = np.concatenate((fld[:n * nx, :], fld[n * nx:2 * (n * nx)]), axis=1)
    tmp = fld[2 * (n * nx) + nx:, ::-1]  # Reverse west hemisphere for proper alignment
    western = np.concatenate((tmp[2::n, :].T, tmp[1::n, :].T, tmp[0::n, :].T))

    # Handle the Arctic region separately
    arctic = fld[2 * (n * nx):2 * (n * nx) + nx, :]

    # Split Arctic region into east and west components
    arctice = np.concatenate((np.triu(arctic[::-1, :nx // 2].T), np.zeros((nx // 2, nx))), axis=1)
    mskr = np.tri(nx // 2)[::-1, :]  # Mask for Arctic region adjustments
    arcticw = np.concatenate((arctic[0:nx // 2, nx:nx // 2 - 1:-1].T,
                              arctic[nx // 2:nx, nx:nx // 2 - 1:-1].T * mskr,
                              np.triu(arctic[nx:nx // 2 - 1:-1, nx:nx // 2 - 1:-1]),
                              arctic[nx:nx // 2 - 1:-1, nx // 2 - 1::-1] * mskr), axis=1)

    # Merge hemispheres and Arctic regions based on chosen centering
    if center == 'Pacific':
        gfld = np.concatenate((np.concatenate((eastern, arctice)),
                               np.concatenate((western, arcticw))), axis=1)
    else:  # Default: Atlantic-centered
        gfld = np.concatenate((np.concatenate((western, arcticw)),
                               np.concatenate((eastern, arctice))), axis=1)

    return gfld


def simple_plot_flat_data(data_to_plot, cmap='viridis', title=None, unit=None):
    """
    Plots a 2D data field using Matplotlib.

    Parameters:
        - data_to_plot (np.ndarray): The 2D array to visualize.
        - cmap (str, optional): Colormap for the plot. Default is 'viridis'.
        - title (str, optional): Title of the plot.
        - unit (str, optional): Unit of the data (used for colorbar label).

    Returns:
        - None (Displays the plot)
    """

    plt.figure(figsize=(12, 6))
    plt.imshow(data_to_plot, cmap=plt.get_cmap(cmap), origin='lower')

    # Add colorbar
    cbar = plt.colorbar()
    if unit:
        cbar.set_label(unit)

    # Axis labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Add title if provided
    if title:
        plt.title(title)

    plt.show()


def transp_tiles(data):
    """
    Transpose specific tiles in the input data array.

    Parameters:
        - data (numpy array): 2D array representing tile data.

    Returns:
        - numpy array: Transformed data with specific tiles transposed.
    """
    nx = data.shape[1]  # Number of columns (x-dimension)
    ny = data.shape[0]  # Number of rows (y-dimension)

    # Extract the bottom part of the data and flip it horizontally
    tmp = data[7 * nx:, ::-1]

    # Rearrange the extracted section by taking every third row in reverse order
    transpo = np.concatenate((tmp[2::3, :].transpose(),
                              tmp[1::3, :].transpose(),
                              tmp[0::3, :].transpose()))

    # Combine the original top section with the transformed tiles
    data_out = np.concatenate((data[:7 * nx], np.flipud(transpo[:, :nx]), np.flipud(transpo[:, nx:])))

    return data_out


def plot_tiles(data, tsz, title=None):
    """
    Plot the tiles from the given data array.

    Parameters:
        - data (numpy array): 2D array representing tile data.
        - tsz (int): Size of each tile.
        - title (str, optional): Title for the plot.

    Returns:
        - None (Displays the plot)
    """

    #### Initialize tile positions ####
    iid = [4, 3, 2, 4, 3, 2, 1, 1, 1, 1, 0, 0, 0]  # Row indices for tile placement
    jid = [0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4]  # Column indices for tile placement
    tid = 0  # Tile index counter

    # Transpose specific tiles (tiles 8 to 13)
    data2d = transp_tiles(data)

    #### Plot tiles ####
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(5, 5, wspace=.05, hspace=.05)  # Create a grid for subplot arrangement

    for i in range(len(iid)):
        ax = fig.add_subplot(gs[iid[i], jid[i]])  # Create subplot for each tile

        # Display the tile image, applying a transpose for the last six tiles
        if i >= 7:
            ax.imshow(data2d[tid:tid + tsz].T, origin='lower')
        else:
            ax.imshow(data2d[tid:tid + tsz], origin='lower')

        tid += tsz  # Move to the next tile

        # Remove axis ticks and labels for a cleaner look
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Add title if provided
    if title:
        fig.suptitle(title)

    plt.show()


def plot_flat_data(longitudes, latitudes, data, variable_name, colorbar_title=None, unit=None, extent=None,
                   depth_value=None, time_value=None, cmap='coolwarm', min_max=None):
    """
    Plots a selected oceanographic variable (DIC, Alk, U and V velocities) on a geographical map.

    Parameters:
        - longitudes (numpy array): 1D array of longitude values.
        - latitudes (numpy array): 1D array of latitude values.
        - data (numpy array): 2D array of data (latitude, longitude).
        - variable_name (str): Name of the variable to display on the plot.
        - colorbar_title (str, optional): Custom title for the colorbar.
        - unit (str, optional): Unit of the variable.
        - extent (list, optional): Create a red box around the region of interest.
        - depth_value (float, optional): Depth of the data.
        - time_value (str, optional): Date of the data.
        - cmap (str, optional): Colormap for the plot. Default is 'coolwarm'.
        - min_max (list, optional): Modify the scale of values/colorbar with a minimum and maximum .


    Returns:
        - None (Displays the plot)
    """

    # Create the figure with a geographical projection
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the data using pcolormesh
    if min_max is None:
        img = ax.pcolormesh(longitudes, latitudes, data, shading="auto", cmap=cmap, transform=ccrs.PlateCarree())
    else:
        img = ax.pcolormesh(longitudes, latitudes, data, shading="auto", cmap=cmap, transform=ccrs.PlateCarree(),
                        vmin=min_max[0], vmax=min_max[1])

    # Adjust the extent of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())

    # Add coastline features
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)

    # Format the colorbar label with the unit if provided
    if colorbar_title:
        cbar_label = f"{colorbar_title}"
    else:
        cbar_label = f"{variable_name}"
    if unit:
        cbar_label += f" ({unit})"

    # Add a colorbar
    plt.colorbar(img, ax=ax, orientation="vertical", label=cbar_label)

    # If a region is provided, plot it
    if extent:
        ax.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
                [extent[2], extent[2], extent[3], extent[3], extent[2]],
                color="red", linestyle='-')

    # Set x and y axis limits (longitude/latitude)
    ax.set_xticks(np.linspace(longitudes.min(), longitudes.max(), num=5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(latitudes.min(), latitudes.max(), num=5), crs=ccrs.PlateCarree())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set the title
    title = f"{variable_name} data"
    if depth_value:
        title += f" at {depth_value} meters"
    if time_value:
        title += f" on the date {time_value}"
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def find_nearest_indices(longitudes, latitudes, target_lon, target_lat):
    """
    Find the (X, Y) indices closest to a given longitude and latitude.

    Parameters:
        - longitudes : np.array 2D containing the longitude values of the grid.
        - latitudes : np.array 2D containing the latitude values of the grid.
        - target_lon : float, target longitude.
        - target_lat : float, target latitude.

    Returns:
        - (y_idx, x_idx) : tuple of indices corresponding to the closest coordinates.
        - (nearest_lon, nearest_lat) : tuple of actual closest coordinates.
    """
    # Compute the Euclidean distance between each grid point and the target coordinates
    distance = np.sqrt((longitudes - target_lon) ** 2 + (latitudes - target_lat) ** 2)

    # Find the index of the minimum distance value
    y_idx, x_idx = np.unravel_index(np.argmin(distance), distance.shape)

    # Retrieve the actual coordinates of the closest point
    nearest_lon = longitudes[y_idx, x_idx]
    nearest_lat = latitudes[y_idx, x_idx]

    return (y_idx, x_idx), (nearest_lon, nearest_lat)


def interpolate_missing_values(grid_data, grid_name, sub_y=189, sub_x=270):
    """
    Interpolates missing values in smaller patches to reduce memory usage.

    Parameters:
        - grid_name (str): Name of the grid.
        - grid_data (numpy array): Data grid (nz, ny, nx).
        - sub_y (int): Height of sub-block.
        - sub_x (int): Width of sub-block.

    Returns:
        - None (saves the interpolated data grid to new NetCDF files).
    """
    # Define the filename
    filename = f"{grid_name}_inter.npy"

    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
        return

    # Copy the data to avoid modifying the original
    grid_interpolated = np.copy(grid_data)
    z, y, x = grid_data.shape

    for depth in range(z):
        for y_start in tqdm(range(0, y, sub_y)):
            for x_start in tqdm(range(0, x, sub_x)):
                # Define the end indices
                y_end = min(y_start + sub_y, y)
                x_end = min(x_start + sub_x, x)

                # Extract the sub-block
                data_block = grid_interpolated[depth, y_start:y_end, x_start:x_end]

                # Create coordinate grids
                X_block, Y_block = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

                # Valid points (non-zero values)
                mask_valid = data_block != 0
                X_valid, Y_valid, data_valid = X_block[mask_valid], Y_block[mask_valid], data_block[mask_valid]

                # Missing points (values equal to 0)
                mask_missing = data_block == 0
                X_missing, Y_missing = X_block[mask_missing], Y_block[mask_missing]

                if len(X_valid) == 0 or len(
                        X_missing) == 0:  # If there are no valid points or nothing to interpolate, skip this layer
                    continue

                # Convert data to sine and cosine to handle cyclic values (-180° / 180°)
                Z_valid_sin = np.sin(np.radians(data_valid))
                Z_valid_cos = np.cos(np.radians(data_valid))

                # Nearest neighbor interpolation
                sin_interp_nearest = spi.griddata((X_valid, Y_valid), Z_valid_sin, (X_missing, Y_missing),
                                                  method="nearest")
                cos_interp_nearest = spi.griddata((X_valid, Y_valid), Z_valid_cos, (X_missing, Y_missing),
                                                  method="nearest")

                # Cubic interpolation
                sin_interp = spi.griddata((X_valid, Y_valid), Z_valid_sin, (X_missing, Y_missing), method="cubic")
                cos_interp = spi.griddata((X_valid, Y_valid), Z_valid_cos, (X_missing, Y_missing), method="cubic")

                # Replace NaN values from cubic interpolation with nearest neighbor results
                sin_interp = np.where(np.isnan(sin_interp), sin_interp_nearest, sin_interp)
                cos_interp = np.where(np.isnan(cos_interp), cos_interp_nearest, cos_interp)

                # Reconstruct interpolated values
                data_interp_corrected = np.degrees(np.arctan2(sin_interp, cos_interp))

                # Update the sub-block with interpolated values
                data_block[mask_missing] = data_interp_corrected

                # Ensure values remain within the [-180, 180] range
                data_block = np.mod(data_block + 180, 360) - 180

                # Update the interpolated data in the main grid
                grid_interpolated[depth, y_start:y_end, x_start:x_end] = data_block

    # Save interpolated grid to Numpy file
    np.save(filename, grid_interpolated)


def create_netcdf(var_name, dtime, var_data, time, depth, lon, lat, latitude='Latitude',
                  longitude='Longitude', date=None, var_units=None, var_long_name=None, depth_comment=None):
    """
    Create a NetCDF file with specified variable and metadata.

    Parameters:
        - var_name (str): Name of the main variable.
        - dtime (str): Date time of data.
        - var_data (numpy array): 3D array (depth, lat, lon) of the variable.
        - time (numpy array): 1D array of time values.
        - depth (numpy array): 1D array of depth values.
        - lon (numpy array): 2D array of longitude values.
        - lat (numpy array): 2D array of latitude values.
        - latitude (str, optional): Name of the latitude variable.
        - longitude (str, optional): Name of the longitude variable.
        - date (str, optional): Start date of the data.
        - var_units (str, optional): Units for the main variable.
        - var_long_name (str, optional): Long name for the main variable.
        - depth_comment (str, optional): Comment to define the depth position of the grid cell

    Returns:
        - None (Creates a NetCDF file with the specified data and metadata).
    """
    # Define the filename
    filename = f"{var_name}_{dtime}.nc"

    # Check if the file already exists → remove it to overwrite cleanly
    if os.path.exists(filename):
        print(f"File {filename} already exists. Overwriting...")
        os.remove(filename)

    # Create NetCDF file
    dataset = Dataset(filename, "w", format="NETCDF4")

    # Create dimensions
    dataset.createDimension("Time", 1)  # Time dimension : 1 time step for each file
    dataset.createDimension("Depth", var_data.shape[0])  # Depth dimension
    dataset.createDimension("Y", var_data.shape[1])  # Latitude dimension
    dataset.createDimension("X", var_data.shape[2])  # Longitude dimension

    # Create coordinate variables
    lat_var = dataset.createVariable(f"{latitude}", "f8", ("Y", "X"))
    lon_var = dataset.createVariable(f"{longitude}", "f8", ("Y", "X"))
    time_var = dataset.createVariable("Time", "f8", ("Time"))
    depth_var = dataset.createVariable("Depth", "f8", ("Depth"))

    # Create main variable with appropriate dimensions (grid)
    var = dataset.createVariable(var_name, "f8", ("Time", "Depth", "Y", "X"))

    # Assign values to variables
    lat_var[:] = lat  # Latitude values
    lon_var[:] = lon  # Longitude values
    time_var[:] = time
    depth_var[:] = depth
    var[:] = var_data

    # Add metadata
    lat_var.units = "degrees_north"
    lon_var.units = "degrees_east"
    depth_var.units = "meters"
    depth_var.comment = depth_comment
    time_var.comment = date  # Start date of the data
    time_var.long_name = "Start date of the data in nanoseconds since 1970-01-01 00:00:00 UTC"
    var.units = var_units
    var.long_name = var_long_name
    var.FillValue = -9.9999998e+22  # Missing value indicator

    # Add global metadata
    dataset.description = f"{var_name} data with latitude/longitude coordinates on a C-grid, depth and time"

    # Close the NetCDF file
    dataset.close()

    print(f"File {filename} created successfully!")


def extract_data_at_depth(var, dtime, depth, latitude='Latitude', longitude='Longitude'):
    """
    Load NetCDF files and extract data at a specified depth.

    Parameters:
        - var (str): Variable name (e.g., DIC, ALK, UVEL, VVEL).
        - dtime (str): Timestamp or file identifier.
        - depth (int): Depth level to extract.
        - latitude (str): Name of the latitude variable.
        - longitude (str): Name of the longitude variable.

    Returns:
        - None (saves the extracted data at the specified depth to a new NetCDF file).
    """
    # Check if the file already exists → remove it to overwrite cleanly
    if os.path.exists(f"{var}_{dtime}_{depth}.nc"):
        print(f"File {var}_{dtime}_{depth}.nc already exists. Overwriting...")
        os.remove(f"{var}_{dtime}_{depth}.nc")

    # Load NetCDF file
    ds = xr.open_dataset(f"{var}_{dtime}.nc").set_coords([f"{latitude}", f"{longitude}"])

    # Select data at the specified depth
    data_at_depth = ds[var].isel(Depth=depth)

    # Save extracted data
    data_at_depth.to_netcdf(f"{var}_{dtime}_{depth}.nc")

    print(f"{var} data extracted and saved for depth {depth}.")


def crop_var(var, var_name, coord_box, depth=None):
    """
    Extract and crop data around a specified geographic bounding box,
    and return the corresponding indices of latitude and longitude for the cropped region.

    Parameters:
        var (xarray.Dataset): Dataset containing the variable to crop.
        var_name (str): Name of the variable.
        coord_box (list or tuple): [lon_min, lon_max, lat_min, lat_max].
        depth (int, optional): Depth level for which data is to be cropped.

    Returns:
        tuple: (var_cropped, lat_indices, lon_indices)
            var_cropped: 2D NumPy array of cropped data.
            lat_indices: Indices corresponding to the cropped latitudes.
            lon_indices: Indices corresponding to the cropped longitudes.
    """
    lon_min, lon_max, lat_min, lat_max = coord_box

    # Get the latitude and longitude variables
    lon = var["Longitude"]
    lat = var["Latitude"]

    # Create a boolean mask for the desired geographic region
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    if depth != None:
        # Apply the mask to the variable at a given depth layer (assuming time dimension, e.g., Time=0)
        var_masked = var[f"{var_name}"].isel(Time=0).isel(Depth=depth).where(mask)
    else:
        # Apply the mask to the variable (assuming time dimension, e.g., Time=0)
        var_masked = var[f"{var_name}"].isel(Time=0).where(mask)

    # Convert to NumPy arrays
    var_values = var_masked.values

    # Identify valid (non-NaN) data
    valid_mask = ~np.isnan(var_values)
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError("No valid data found in the specified region.")

    # Get cropping bounds (start and end indices)
    row_start, row_end = np.where(rows)[0][[0, -1]]
    col_start, col_end = np.where(cols)[0][[0, -1]]

    # Crop the arrays to the bounding box with valid data
    var_cropped = var_values[row_start:row_end+1, col_start:col_end+1]

    # Get the corresponding indices for latitudes and longitudes
    lat_indices = np.arange(row_start, row_end+1)
    lon_indices = np.arange(col_start, col_end+1)

    return var_cropped, lat_indices, lon_indices


def compute_dpco2_sensitivity(dic, alk, salt, sst, rho_water=1.028):
    """
    Compute the pCO2 sensitivity to DIC and ALK changes fyor given 2D inputs.

    Parameters:
        - dic, alk, salt, sst (np.ndarray): 2D arrays of DIC, ALK, salinity, and temperature.
        - rho_water (float): Seawater density (default = 1.028 kg/m³).

    Returns:
        - dpco2_ov_ddic (np.ndarray): Sensitivity of pCO2 to DIC.
        - dpco2_ov_dalk (np.ndarray): Sensitivity of pCO2 to ALK.
    """
    # Base pCO2
    res = pyco2.sys(
        par1=dic / rho_water,
        par2=alk / rho_water,
        par1_type=2, par2_type=1,
        salinity=salt + 35,
        temperature=sst
    )
    pco2_ref = res['pCO2']  # In µatm

    # Perturbation values
    delta_dic = 10  # In µmol C/kg
    delta_alk = 10  # In µmol ALK/kg

    # Perturbed pCO2 in response to DIC and ALK changes
    pco2_dic = pyco2.sys(
        par1=(dic + delta_dic) / rho_water,
        par2=alk / rho_water,
        par1_type=2, par2_type=1,
        salinity=salt + 35,
        temperature=sst
    )['pCO2']

    pco2_alk = pyco2.sys(
        par1=dic / rho_water,
        par2=(alk + delta_alk) / rho_water,
        par1_type=2, par2_type=1,
        salinity=salt + 35,
        temperature=sst
    )['pCO2']

    # Sensitivities
    dpco2_ov_ddic = (pco2_dic - pco2_ref) / delta_dic  # In µatm / (µmol C/kg)
    dpco2_ov_dalk = (pco2_alk - pco2_ref) / delta_alk  # In µatm / (µmol ALK/kg)

    return dpco2_ov_ddic, dpco2_ov_dalk


