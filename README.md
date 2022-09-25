# JME-Fairness

This repository contains the implementation for paper: [Joint Multisided Exposure Fairness for Recommendation](https://arxiv.org/abs/2205.00048) (SIGIR 2022).



## Code Overview
# Folder
The [`data`](./data) contains the datasets we used from [here](https://grouplens.org/datasets/movielens/).

The [`saved_model`](./saved_model) contains the pre-trained model from [here](https://github.com/dvalcarce/evalMetrics).

# File
The [`read_data.py`](./read_data.py) contains the data reading and preprocessing.
The [`Disparity_Metrics.py`](./Disparity_Metrics.py) contains the implementation of our proposed JME-Fairness metrics.
The [`run_metric.py`](./run_metric.py) outputs the output values for different JME-Fairness metrics.

## Run code
```
python run_metric.py
```


## Citation
If you find this code or idea useful, please cite our work:
```
@inproceedings{wu2022joint,
  title={Joint Multisided Exposure Fairness for Recommendation},
  author={Wu, Haolun and Mitra, Bhaskar and Ma, Chen and Diaz, Fernando and Liu, Xue},
  booktitle={SIGIR},
  publisher = {{ACM}},
  year={2022}
}
```

## Contact
If you have any questions, feel free to contact us through email (haolun.wu@mail.mcgill.ca) or Github issues. Enjoy!
