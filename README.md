# Meta Learning for Code Summarization
A meta learning based approach for code summarization, is an attempt to combine the strengths of individual code summarization models through meta learning. 
See our [paper](https://arxiv.org/abs/2201.08310) for details:
```
@article{rauf2022meta,
  title={Meta Learning for Code Summarization},
  author={Rauf, Moiz and Pad{\'o}, Sebastian and Pradel, Michael},
  journal={arXiv preprint arXiv:2201.08310},
  year={2022}
}
```

This repository provides implementation for two neural network based models and one feature based model. 
This repository contains the following folder:
1. models: consisting of implementation for neural meta models and feature based model.
2. Scripts folder: containing utility scripts for computing output from meta models.


### Training/Testing Models
We provide implementation for neural-meta models (Transformer and BiLSTM based) in addtion to feature based model. To perform training and evaluation, first go the scripts directory associated with the target model. 

```
$ cd  models/MODEL_NAME
```
Where, choices for MODEL_NAME are ["feature_model", "neural_models"].

To train/evaluate a model, run:

```
$ bash run.sh 
```
#### Generated log files

While training and evaluating the models, a list of files are generated inside a `results` directory. The files are as follows.

- **MODEL_NAME_epoch_id.pt**
  - Model files containing the parameters of the model per epoch.
- **Config.txt**
  - Configuration file with hyperparameter details.
- **predictions.json**
  - The predictions file from the code.
In addition to the files, a further `res` directory is created in order to save output of meta model and resultant bleu score. The following files are created in that folder. 
- **meta_summ.txt**
  - File containing candidate summaries
- **corpus_bleu.txt**
  - File containing corpus BLEU.

#### Requirements
The code requires Linux and Python 3.6 or higher. It also requires installing PyTorch version 1.3 or higher. We additionally require NLTK and  Scikit-learn. CUDA is strongly recommended for speed, but not necessary.

