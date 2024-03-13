# SEGDet - ML Reproductibility Challenge 2024

>📋  Optional: include a graphic explaining main result (+bibtex entry, link to demos, blog posts and tutorials?)

## Installation

Python packages required (can be installed via pip or conda):

- python >= 3.6.1

>📋  Verify that the Python version is correct.

Please refer to [doc/INSTALL.md](doc/INSTALL.md) for more details on how to install the required packages and dependencies.



More details on how to install the required packages and dependencies can be found in [doc/INSTALL.md](doc/INSTALL.md).

Details on how to download the datasets used in this project can be found in [doc/DATASETS.md](doc/DATASETS.md).

>📋  Anything else required to set up the environment?

Download INSERT HERE dataset:

```python
print("Hello World")
```

If desired, use the same seed used for our results below:

```python
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

## Training

To train the model in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Example commands on how to train, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on [MS COCO Val 2017](https://cocodataset.org/#download), run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Evaluate the trained models on benchmarks reported, give commands that produce the results just below.

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on SOMETHING using parameters x,y,z. 

## Results

Our model achieves the following performance on :

### [Image Classification on SOMETHING](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Don't forget to link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

Lists of interesting results to reproduce in RTMDet:

- Comparison with RTMDet on the number of parameters, FLOPS, latency, and accuracy on COCO
val2017 set (300 iterations, Table 2 and 3)
- Architecture performance comparaison with reproduced model (Table 5)
- Reproduce some results only shown for RTMDet-R (we reproduce RTMDet-ins only)?

## Citation

If you use this code in your research, please cite the following:

@inproceedings{lim2019fast,
  title={RTMDet MLRP2024},
  author={Pierre-Luc Asselin, Vincent Coulombe, William Guimont-Martin, William Larrivée-Hardy},
  year={2024}
}

## Contributing

We want to make contributing to this project as easy and transparent as
possible.

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

### License
By contributing, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
