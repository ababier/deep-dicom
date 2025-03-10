# DeepDicom

_DeepDicom_ provides tools to process DICOM files for deep learning research. It includes tools to extract images and metadata from DICOM files, standardize image dimensions, and standardize structure names. The repository also includes a training pipeline for a PyTorch UNet model to do dose prediction. 

This repository should support DICOMs from any radiotherapy treatment planning system. This includes clinical data publicly available DICOM from the Cancer Imaging Archive (e.g., [Pancreatic-CT-CBCT-SEG](https://www.cancerimagingarchive.net/collection/pancreatic-ct-cbct-seg/)).

## Table of Contents
- [What This Code Does](#what-this-code-does)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Running the Code](#running-the-code)

## What This Code Does
DeepDicom consists of four modules:
1. **Interface**: Contains the `Case` class to organize DICOM data.
2. **Dicom Extraction**: Extracts images and metadata from DICOM files and stores them in an efficient format.
3. **Prediction**: Standardizes data from `Case` objects, creates training samples, and defines data splits.
4. **Model**: Contains the `Trainer` class that trains a UNet model for dose prediction using PyTorch.

## Prerequisites
- Linux
- Python 3.10.12
- NVIDIA GPU with CUDA and CuDNN (recommended)


## Getting Started
1. **Create a virtual environment and activate it:**
    ```bash
    virtualenv -p python3 deep-dicom
    source deep-dicom/bin/activate
    ```
2. **Clone the repository and install dependencies:**
    ```bash
    git clone https://github.com/ababier/deep-dicom
    cd deep-dicom
    pip3 install -r requirements.txt
    ```


## Running the Code
If everything is set up correctly, run the main script to start processing DICOM files, generate training samples, and train the model.

Run the following command in your virtual environment:
```bash
python3 main.py
```
This will:
- Extract DICOM data and organize it into `Case` objects.
- Standardize images, voxel spacing, and structure names.
- Create train/validation/test data splits.
- Train a UNet model for dose prediction, logging progress with TensorBoard.

### Additional Notes
- **Data Splits:**  
  The code uses dataset splits. The training, validation, and test IDs are saved and re-used for consistency.
  
- **Model Training:**  
  The Trainer class uses gradient checkpointing and AMP (Automatic Mixed Precision) for efficient training, along with an exponential learning rate scheduler.
  
- **TensorBoard Logging:**  
  Training metrics and image visualizations are logged. Launch TensorBoard by running:
  ```bash
  tensorboard --logdir=runs/ --port=6006 --bind_all
  ```

- **Customization:**  
  You can adjust hyperparameters (e.g., batch size, learning rate) in the Trainer class, and modify the UNet architecture or data processing as needed.

Happy coding and deep learning with DICOM data!
