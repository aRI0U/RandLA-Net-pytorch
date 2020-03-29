# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236).

## Preparation

1. Clone this repository

   ```sh
   git clone https://github.com/aRI0U/RandLA-Net-pytorch.git
   ```

2. Install all Python dependencies (TODO write `requirements.txt`)

  ```sh
    cd RandLA-Net-pytorch
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

## Citation

This work implements the work presented in [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](http://arxiv.org/abs/1911.11236).

The original implementation (in TensorFlow 1) can be found [here](https://github.com/QingyongHu/RandLA-Net).

To cite the original paper:
  ```
  @article{RandLA-Net,
    arxivId = {1911.11236},
    author = {Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
    eprint = {1911.11236},
    title = {{RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds}},
    url = {http://arxiv.org/abs/1911.11236},
    year = {2019}
  }
  ```


## TODOs

make data.py support different input dims
make cpu work

optimization:
- replace read_ply mmap_mode
- limit memory usage
- see whether cross entropy with dtype uint8 possible
