# Training

Store dataset at the same location as `train.py`. Make suitable changes in `train.py`:

```
dataset_name = 'forehead-v1-labeled'
suffix = f'_adaface' # either leave empty or 
                     # always begin with _ (use when you repeat a dataset but with some parameter(s) changed)

test_dataset_name = 'cross_database'
```

To load from a pretrained model, set `load_from_pretrained = True` and specify the model path in `saved_weights_path` variable.


```
python train.py
```

# Acknowledgements

- [GitHub: Bharadwaj et. al. (WACV 2022)](https://github.com/rohit901/ForeheadCreases/)
- [AdaFace Loss](https://github.com/mk-minchul/AdaFace/blob/master/head.py)