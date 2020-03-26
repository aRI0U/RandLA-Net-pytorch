# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

## Preparation

1. Clone this repository

   ```sh
   git clone https://github.com/aRI0U/RandLA-Net-pytorch.git
   ```

2. Install the dependencies

3. Download the dataset and prepare it:

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

architecture:
- change mlp to convolutions

change knn method
