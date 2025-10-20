"""Band normalization utilities"""
import numpy as np


def normalize_band(band_data, percentile_clip=2):
    """
    Normalize band data to 0-1 range with percentile clipping.
    This helps improve visualization by removing outliers.

    Args:
        band_data: numpy array containing band data
        percentile_clip: percentile value for clipping (default: 2)

    Returns:
        numpy array with normalized data in range [0, 1]
    """
    p_low = np.percentile(band_data, percentile_clip)
    p_high = np.percentile(band_data, 100 - percentile_clip)
    normalized = np.clip((band_data - p_low) / (p_high - p_low), 0, 1)
    return normalized
