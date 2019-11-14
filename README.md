# Xtreme Materials
>A Repository for Lifetime Prediction of High Temperature Alloy Materials

In this repo, we will develop machine learning models to predict the lifetime of alloy materials for high temperature applications.  

## Focus areas:

1. Hyperparameter optimization (grid search, random search, Bayesian, MCC)
2. Getting posterior distribution using MCMC


## TODO list:

- Data collection
- Preprocessing
- Model generation
- Model selection
- Model averaging
    
## Installing / Getting started

To use the codes in this package, you need the following packages:

```shell
numpy>=1.12
sklearn>=0.20.1
matplotlib>=3.1
pandas>=0.25.1
```

After, you have all the dependencies installed, download the code from the bitbucket repo using:

```shell
git clone ssh://git@stash.pnnl.gov:7999/~visw924/xmat-pnnl.git
```

Though you can run `setup.py` to install the package using:

```shell
python setup.py install 
```

I would recommend adding the following line in your `PYTHONPATH`:

```shell
export PYTHONPATH="path/to/xmat_pnnl_code:${PYTHONPATH}"
```

## Features 

What's all the bells and whistles this code can perform:
* Linear regression 

## How to use different modules

The organization of the xmat-pnnl is as follows:

![Tree view of the file system](images/code_tree.png)

* data_processing: contains processing of individual dataset and then saves the resulting data in a dictionary format in a `.npy` file.
* xmat_pnnl_code: contains the code or wrapper for individual algorithm.
* xmat_pnnl_data: contains the raw data in excel format
* xmat_pnnl_model: models are run and the results are stored in this folder

In the below, I will mainly document all the code or wrapper for different algorithm and how to use them.

