# GANcon-pytorch

****[Note]**. This is a _personal_ reimplementation of the GANcon paper (GANcon: protein contact map prediction with deep generative adversarial network) on a _third-party_ dataset. This repository was one of the course projects and the dataset was provided by the course instructor, so I have no permission to distribute the data. If you encounter any problems (especially with the original data, the original checkpoint, and issues with the original paper), I would suggest asking the original authors for help.**

## Requirements

Here are the recommended environment settings of this repository.

- Python >= 3.7.9
- PyTorch >= 1.7.1
- Numpy >= 1.19.2
- tqdm

## Datasets

In this repository, we only provide the dataset with input feature of size `L * L * 441`, where 441 is the feature channels, and ground-truth labels of size `L * L * 10`, where 10 is the output classes, because we have further divided the distance between amino acids into 10 classes and form a multi-categorical classification problem, instead of the tradition binary classification problem.

If you want to use other datasets, you may need to write `dataset.py` by yourself. Or you can create an issue, and we will implement the dataset you provided when the maintainers are free.

If you are using datasets that have not been divided into training set, validation set and testing set yet, you can use our tools to preprocess the dataset. See [docs/preprocessing.md](docs/preprocessing.md)

## Configurations

To train or test the model, you need to create a configuration file in yaml format. We have provided an example configuration file, see `configs/default.yaml`. To customize the configuration file, see [docs/configs.md](docs/configs.md) for details.

## Training

Simply executing the following commands.

```bash
python train.py --cfg [Config File]
```

- `[Config File]` is the path to the configuration file, default: `configs/default.yaml`.

## Testing

Simply executing the following commands.

```bash
python test.py --cfg [Config File]
```

- `[Config File]` is the path to the configuration file, default: `configs/default.yaml`.
- For our provided dataset, we provide our own criterion that measures the accuracy of different classes, which is implemented in `utils/criterion.py`. For other dataset, we only provide the original loss criterion.

## To-dos

- [ ] Pretrain models
- [ ] Dataset downloading
- [ ] Other dataset supporting
- [ ] Demo presentation

## References

Yang, Hang, et al. "GANcon: protein contact map prediction with deep generative adversarial network." IEEE Access 8 (2020): 80899-80907.

## Citation

```bibtex
@article{yang2020gancon,
  title =        {GANcon: protein contact map prediction with deep generative adversarial network},
  author =       {Yang, Hang and Wang, Minghui and Yu, Zhenhua and Zhao, Xing-Ming and Li, Ao},
  journal =      {IEEE Access},
  volume =       {8},
  pages =        {80899--80907},
  year =         {2020},
  publisher =    {IEEE}
}

@misc{fang2021gancon,
  author =       {Hongjie Fang, Zhanda Zhu, Peishen Yan and Hao Yin},
  title =        {GANcon re-implementation in PyTorch},
  howpublished = {\url{https://github.com/Galaxies99/GANcon-pytorch}},
  year =         {2021}
}
```
