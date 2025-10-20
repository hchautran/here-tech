"""Band visualization utilities"""
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from ..utils.normalization import normalize_band


def visualize_all_bands(filepath, show_plot=True):
    """
    Visualize all individual bands of a satellite image.

    Args:
        filepath: path to the TIF file
        show_plot: whether to display the plot (default: True)

    Returns:
        matplotlib figure object
    """
    # Read the satellite image data
    with rasterio.open(filepath) as src:
        data = src.read()  # Shape: (bands, height, width)
        num_bands = data.shape[0]

    # Create figure with subplots for each band
    cols = min(4, num_bands)
    rows = (num_bands + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    # Flatten axes array for easier indexing
    if num_bands == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_bands > 1 else [axes]

    filename = os.path.basename(filepath)

    # Plot each band
    for i in range(num_bands):
        ax = axes[i]
        band_img = normalize_band(data[i])

        # Display the band
        im = ax.imshow(band_img, cmap='viridis')
        ax.set_title(f'Band {i+1}')
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(num_bands, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Individual Bands: {filename}', fontweight='bold', fontsize=20)
    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig


def visualize_band_combinations(filepath, combinations='all', show_plot=True):
    """
    Visualize various band combinations for satellite imagery analysis.

    Combinations include:
    - True color: RGB representation (B4, B3, B2)
    - False color: Vegetation analysis (B8, B4, B3)
    - False color urban: Urban areas (B12, B11, B4)
    - Natural color: Enhanced natural color
    - NDVI: Vegetation index
    - NDWI: Water index
    - NDSI: Snow index
    - Moisture index: Soil moisture
    - SWIR: Shortwave infrared

    Args:
        filepath: path to the TIF file
        combinations: 'all' or list of specific combinations (default: 'all')
        show_plot: whether to display the plot (default: True)

    Returns:
        matplotlib figure object
    """
    # Read the satellite image data
    with rasterio.open(filepath) as src:
        data = src.read().astype(float)  # Shape: (bands, height, width)
        num_bands = data.shape[0]

    filename = os.path.basename(filepath)
    visualizations = []

    # 1. True Color (Natural RGB) - B4, B3, B2
    if num_bands >= 3:
        rgb = np.stack([
            normalize_band(data[2]),  # Red (B4)
            normalize_band(data[1]),  # Green (B3)
            normalize_band(data[0])   # Blue (B2)
        ], axis=-1)
        visualizations.append(('True Color\n(B4, B3, B2)', rgb, None, 'RGB natural color'))

    # 2. False Color (Vegetation) - B8, B4, B3
    if num_bands >= 4:
        false_color = np.stack([
            normalize_band(data[3]),  # NIR (B8) -> Red
            normalize_band(data[2]),  # Red (B4) -> Green
            normalize_band(data[1])   # Green (B3) -> Blue
        ], axis=-1)
        visualizations.append(('False Color\n(B8, B4, B3)', false_color, None, 'Vegetation appears red'))

    # 3. Highlight Optimized Natural Color
    if num_bands >= 4:
        natural = np.stack([
            normalize_band(data[3]),  # NIR (B8)
            normalize_band(data[2]),  # Red (B4)
            normalize_band(data[1])   # Green (B3)
        ], axis=-1)
        visualizations.append(('Natural Color Enhanced\n(B8, B4, B3)', natural, None, 'Enhanced natural color'))

    # 4. NDVI - Normalized Difference Vegetation Index
    if num_bands >= 4:
        # NDVI = (NIR - Red) / (NIR + Red)
        nir = data[3]
        red = data[2]
        ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
        visualizations.append(('NDVI\n(B8-B4)/(B8+B4)', ndvi, 'RdYlGn', 'Vegetation health'))

    # 5. False Color Urban - B12, B11, B4
    if num_bands >= 7:
        urban = np.stack([
            normalize_band(data[6]),  # SWIR2 (B12)
            normalize_band(data[5]),  # SWIR1 (B11)
            normalize_band(data[2])   # Red (B4)
        ], axis=-1)
        visualizations.append(('False Color Urban\n(B12, B11, B4)', urban, None, 'Urban areas highlighted'))

    # 6. Moisture Index - (B8A - B11) / (B8A + B11)
    if num_bands >= 6:
        b8a = data[4]
        b11 = data[5]
        moisture = np.where((b8a + b11) != 0, (b8a - b11) / (b8a + b11), 0)
        visualizations.append(('Moisture Index\n(B8A-B11)/(B8A+B11)', moisture, 'YlGnBu', 'Soil moisture'))

    # 7. SWIR - B12, B8A, B4
    if num_bands >= 7:
        swir = np.stack([
            normalize_band(data[6]),  # SWIR2 (B12)
            normalize_band(data[4]),  # NIR (B8A)
            normalize_band(data[2])   # Red (B4)
        ], axis=-1)
        visualizations.append(('SWIR\n(B12, B8A, B4)', swir, None, 'Shortwave infrared'))

    # 8. NDWI - Normalized Difference Water Index
    if num_bands >= 4:
        # NDWI = (Green - NIR) / (Green + NIR) for water detection
        green = data[1]
        nir = data[3]
        ndwi = np.where((green + nir) != 0, (green - nir) / (green + nir), 0)
        visualizations.append(('NDWI\n(B3-B8)/(B3+B8)', ndwi, 'Blues', 'Water detection'))

    # 9. NDSI - Normalized Difference Snow Index
    if num_bands >= 6:
        # NDSI = (Green - SWIR) / (Green + SWIR)
        green = data[1]
        swir = data[5]  # B11
        ndsi = np.where((green + swir) != 0, (green - swir) / (green + swir), 0)
        visualizations.append(('NDSI\n(B3-B11)/(B3+B11)', ndsi, 'coolwarm', 'Snow/ice detection'))

    # Create figure with subplots
    num_plots = len(visualizations)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

    # Flatten axes array for easier indexing
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each visualization
    for idx, (title, img_data, cmap, description) in enumerate(visualizations):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Display the image
        if cmap is None:
            # RGB composite
            ax.imshow(img_data)
        else:
            # Single band with colormap
            im = ax.imshow(img_data, cmap=cmap)
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        ax.set_title(title)
        ax.axis('off')

        # Add description text at bottom
        ax.text(0.5, -0.05, description, transform=ax.transAxes,
               ha='center', va='top', fontsize=8, style='italic', color='gray')

    # Hide unused axes
    for idx in range(len(visualizations), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Band Combinations Analysis: {filename}',
                fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig
