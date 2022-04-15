# Gradient estimator for symbolical features
This repository is related to *Stochastic Gradient Descent with gradient estimator for symbolical features*.

## Setup the environment
###  PyTorch environment

Install `conda`

```
conda create -n rBatch python=3.8.8
conda activate rBatch

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1.243 numpy=1.19.2 -c pytorch -y
conda install cudnn=7.6.5 -c anaconda -y
pip install -r requirements.txt
conda install -c conda-forge nodejs -y
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# if the following commands do not succeed, update conda
conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}
conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
conda env config vars set CUDA_HOME=${CONDA_PREFIX}
conda env config vars set CUDA_ROOT=${CONDA_PREFIX}

conda deactivate
conda activate rBatch
```


## Reproduce results

to reproduce an experiment, run

```
 python aTOz.py <dataset> <task> <model_name> <epoch> <batch_size> <k>
```

to reproduce **all** experiments with the `data/data.py` settings, run

```
 python launch.py
```



supported datasets:
- `kdd`
- `forest_cover`
- `adult_income`
- `dont_get_kicked`


## Presented results

All experiments were conducted on a Intel(R) Xeon(R) 2.30GHz core under the Python environment.