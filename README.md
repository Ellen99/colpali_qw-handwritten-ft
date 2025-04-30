# Project Setup and Dataset Information

## Installation Instructions

Follow these steps to set up your environment and install the necessary packages:

1. **Create a Python virtual environment** with `pyenv`:
   ```bash
   pyenv virtualenv 3.11.6 colpali-env
   ```

2. **Activate the virtual environment**:
   ```bash
   pyenv activate colpali-env
   ```

3. **Install the `colpali` package directly from GitHub**:
   ```bash
   pip install git+https://github.com/illuin-tech/colpali.git
   ```

4. **Install the `colpali-engine` package**:
   ```bash
   pip install colpali-engine
   ```

## Dataset Information

This project uses the **IAM Handwriting Database**.


The IAM and LAM dataset files required for this project are available under the Releases section of this repository.
- Go to the Releases page.
- Download the dataset archives (e.g., IAM.zip, LAM.zip).
- Extract the contents.
Place the extracted data/IAM/ folder inside the project root directory.

extract IAM.zip
  - `data/IAM/lines/` — contains individual handwritten text line images.
  - `data/IAM/ascii/lines.txt` — metadata file containing the transcription for each image.

- **Dataset Processing**:
  - Images are loaded in **grayscale**.
  - Transformations (rotation, affine, normalization) are applied to augment the data for better model generalization.
  - Each data sample returns a **transformed image tensor** and its corresponding **text transcription**.

## Project Structure Overview

```
project-root/
├── data/
│   └── IAM/
│       ├── lines/          # Handwritten line images
│       └── ascii/
│           └── lines.txt   # Transcriptions and metadata
│   └── RIMES/   #TODO

├── data_collection/
│   ├── datasets.py         # Dataset class (IAMDataset, RIMESDataset to be added later)
│   └── use_IAM.ipynb # Example usage and visualization
├── colnomic_inference.ipynb  # example inference of image and text for the model we want to fine tune
└── README.md

```

## Quick Start

Example to load and inspect the dataset:

```python
from scripts.datasets import IAMDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05), shear=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = IAMDataset(
    lines_txt_path='../data/IAM/ascii/lines.txt',
    base_image_path='../data/IAM/lines',
    transform=transform
)

print(f"Number of samples: {len(dataset)}")
```
