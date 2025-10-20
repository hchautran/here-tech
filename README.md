# Satellite Image Analysis Project

This project performs analysis on multi-band satellite imagery (GeoTIFF files) using deep learning models including CLIP and DINOv2 for feature extraction and similarity comparison.

## Overview

The project analyzes satellite imagery data with the following capabilities:
- Load and visualize multi-band GeoTIFF files
- Generate various band combinations (True Color, False Color, NDVI, NDWI, etc.)
- Extract features using CLIP and DINOv2 models
- Compare global and local views of satellite images
- Compute similarity scores between image patches

## Dataset

- **Format**: Multi-band GeoTIFF files (`.tif`)
- **Dimensions**: 256x256 pixels
- **Bands**: 12 bands (Sentinel-2 satellite data)
- **Data Type**: uint16
- **Coverage**: Global (100,000 patches)
- **Coordinate System**: Various EPSG codes (UTM zones)

## Project Structure

```
.
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── data/                   # Data loading and processing modules
│   ├── models/                 # Model wrapper classes (CLIP, DINOv2)
│   ├── visualization/          # Visualization utilities
│   └── utils/                  # Helper functions
├── data/                       # Data directory (not tracked)
├── outputs/                    # Output directory
│   ├── images/                 # Generated images
│   └── reports/                # Analysis reports
├── notebooks/                  # Jupyter notebooks
└── scripts/                    # Standalone scripts
```

## Installation

1. Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone <repository-url>
cd here_tech
```

3. Install dependencies using `uv`:
```bash
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

Alternatively, if you prefer using `requirements.txt`:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

To install with development dependencies (Jupyter, etc.):
```bash
uv sync --extra dev
```

You can also use `uv` to run commands directly without activating the virtual environment:
```bash
uv run python your_script.py
```

## Usage

### Data Preparation

Update the data paths in your configuration or scripts:
- `data_dir`: Directory containing TIF files
- `index_file`: CSV file with image metadata (filename, longitude, latitude)

### Running Analysis

#### From Jupyter Notebook
```bash
uv run jupyter notebook analysis_data.ipynb
```

#### From Python Scripts

With activated virtual environment:
```bash
source .venv/bin/activate  
python your_script.py
```

Or directly with `uv run`:
```bash
uv run python your_script.py
```

Example code:
```python
from src.data.loader import load_tif_file
from src.models.clip_model import CLIPModel
from src.visualization.band_viz import visualize_all_bands

# Load a TIF file
data, metadata = load_tif_file("path/to/image.tif")

# Visualize bands
visualize_all_bands("path/to/image.tif")

# Extract CLIP features
clip_model = CLIPModel(device="cuda")
features = clip_model.extract_features("path/to/image.png")
```

## Features

### Band Visualizations

The project supports multiple band combination visualizations:

1. **True Color (B4, B3, B2)**: Natural RGB representation
2. **False Color (B8, B4, B3)**: Vegetation appears red
3. **Natural Color Enhanced**: Enhanced natural color
4. **NDVI**: Normalized Difference Vegetation Index
5. **NDWI**: Normalized Difference Water Index
6. **NDSI**: Normalized Difference Snow Index
7. **Moisture Index**: Soil moisture visualization
8. **SWIR**: Shortwave infrared composite
9. **False Color Urban**: Urban area visualization

### Model Features

- **CLIP (ViT-B/32)**: Visual-language model for image understanding
- **DINOv2 (ViT-L/14)**: Self-supervised vision transformer

### Similarity Analysis

- Compare global and local views of images
- Extract K random patches and compute average similarity
- Support for both CLIP and DINOv2 feature extraction

## Key Functions

### Data Loading
- `load_tif_file()`: Load GeoTIFF files with rasterio
- `load_and_save_tif_as_image()`: Convert TIF to PNG true color image

### Normalization
- `normalize_band()`: Normalize band data with percentile clipping

### Visualization
- `visualize_all_bands()`: Display all individual bands
- `visualize_band_combinations()`: Show various band combinations

### Feature Extraction
- `extract_dinov2_features()`: Extract DINOv2 embeddings
- `compare_two_images()`: Compare similarity between two images
- `clip_similarity()`: Compute CLIP-based similarity scores
- `dinov2_similarity()`: Compute DINOv2-based similarity scores

## Configuration

CUDA device selection:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU device
```

Model parameters:
- CLIP: ViT-B/32 (224x224 input)
- DINOv2: ViT-L/14 (224x224 input)
- Patch size for similarity analysis: 64x64
- Number of patches per image (K): 10

## Results

The analysis provides:
- Geographic distribution of satellite patches
- Band statistics and visualizations
- Similarity scores between global and local views
- Feature embeddings for downstream tasks

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- ~10GB disk space for models and cache

## License

[Add your license here]

## Contributors

[Add contributors here]

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [Sentinel-2 Mission](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
