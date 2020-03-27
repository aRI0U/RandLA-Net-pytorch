# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

## Preparation

1. Clone this repository

   ```sh
   git clone https://github.com/aRI0U/RandLA-Net-pytorch.git
   ```

2. Install all Python dependencies (TODO write `requirements.txt`)

  ```sh
    pip install -r requirements
  ```

3. Install `torch_points` package

  Using `pip`...
  ```sh
    pip install git+https://github.com/nicolas-chaulet/torch-points.git
  ```
  ...or from source
  ```sh
    git clone https://github.com/nicolas-chaulet/torch-points.git
    cd torch_points
    python3 setup.py install
    python3 -m unittest
  ```

4. Download the dataset and prepare it:

   ```sh
   cd RandLA-Net-pytorch/utils
   ./download_semantic3d.sh
   python3 prepare_semantic3d.py # Very slow operation
   ```

## Usage

- Train a model

  ```sh
  python3 train.py
  ```

  Add flag `--gpu` to train the model on GPU instead of CPU.

- Evaluate a model

  ```sh
  python3 test.py
  ```

### Visualization

One can visualize the evolution of the loss with Tensorboard.

On a separate terminal, launch:

  ```sh
  tensorboard --logdir runs
  ```

## TODOs

change knn method

optimization:
- replace read_ply mmap_mode
- replace list by collections.deque in model
