# S3VDC
The Tensorflow implementation of algorithm S3VDC (Simple, Scalable, and Stable Variational Deep Clustering), which is an end-to-end unsupervised deep clustering algorithm. This particular implementation adds the following algorithmic improvements over [VaDE](https://www.ijcai.org/Proceedings/2017/273).
- Initial &gamma;-Training: enable a much better reproducibility (milder volatility) clustering result without the need of any pre-trained model.
- Periodic &beta;-Annealing: improve the disentanglement of the learned latent embeddings.
- Mini-Batch GMM Initialization: make the algorithm scalable to large datasets.
- Inverse Min-Max Transform: avoid NaN losses.

Upon using this repo for your work, please cite [this paper](https://arxiv.org/abs/2005.08047) (or the formally published version if available):
```
@article{cao2020s3vdc,
  title={Simple, Scalable, and Stable Variational Deep Clustering},
  author={Cao, Lele and Asadi, Sahar and Zhu, Wenfei and Schmidli, Christian and Michael, Sj\"{o}berg},
  journal={arXiv preprint arXiv:2005.08047},
  year={2020}
}
```

## Prerequisites
### Prepare virtual environment
Install [Anaconda](https://www.anaconda.com/distribution/) first to avoid breaking your native environment.  
```
conda update conda
conda create --name py36-s3vdc python=3.6
source activate py36-s3vdc
```

### Install dependent libraries
To install the dependent libraries, execute the following commands within the root folder (i.e. `s3vdc`).
```
cd s3vdc
pip install -r requirements.txt
```

### Initialize your workspace
To download the transformed datasets and pretrained models, execute the following in the root folder (i.e. `s3vdc`):
```
./initws.sh
```
Make sure that you have two new folders `datasets` and `models` in the current folder.

## S3VDC Training, Finetuning, and Prediction

We use `<DATASET>` to denote the name of the dataset, which can only be one of `mnist`, `inertial_har`, and `fashion`. 

You can choose the model hyper-parameters and experimental settings in configuration file [`task_<DATASET>/config.json`](task_mnist/config.json). The model is implemented in [`task_<DATASET>/s3vdc.py`](task_mnist/s3vdc.py).

The S3VDC is implemented as Tensorflow estimators. To perform training, finetuning, and prediction on benchmark datasets, follow the instructions below. 


### Train S3VDC from scratch
```
./s3vdc.sh train <DATASET>
```
If the training is interrupted, and you want to resume the training from where it stopped, use the same command above to pick up the training from the latest snapshot. 

The trained model will be stored in folder `task_<DATASET>/model`. To start Tensorboard to monitor the training and evaluation, use:
```
tensorboard --logdir task_<DATASET>/model
```

### Predict with the model trained from scratch
```
./s3vdc.sh pred <DATASET> [<FILE_FORMAT>]
```
The compressed prediction will be exported to folder `task_<DATASET>/model/prediction_<date>_<time>/`. The optional CLI parameter `[<FILE_FORMAT>]` specifies the output format, which can be one of `json` (default), `csv`, and `text`. Note that the JSON format can be directly loaded to BigQuery for further analysis and visualization.

### Predict with the published pretrained model
```
./s3vdc.sh rep <DATASET> [<FILE_FORMAT>]
```
The prediction results can be found in folder `models/<DATASET>/prediction_<date>_<time>/`.


## MISC
To clean up the user-trained models, logs, and predictions, run
```
./clearws.sh
```
To additionally remove the datasets and pretrained models, run
```
./clearws.sh all
```