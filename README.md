# Medical Image Segmentation and Applications

This repository contains all projects for the MISA course, it is mostly multi-atlas image segmentation and deep learning-based segmentation.

## Project Structure

### Directories

- `baseline/`: Contains baseline methods and parameters for image registration.
- `deepLearning/`: Contains deep learning models and training scripts.
- `multiAtlas/`: Contains multi-atlas segmentation methods and parameters.
- `Lab1&2/`: Contains lab exercises and reports for labs 1 and 2.
- `Lab3/`: Contains lab exercises and reports for lab 3.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Marshall-mk/MISA.git
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

#### Deep Learning

To train the deep learning model, navigate to the `deepLearning` directory and run the training script:
```sh
cd final project/deepLearning
jupyter notebook MISA_DEEPL3D.ipynb