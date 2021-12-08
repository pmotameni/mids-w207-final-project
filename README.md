# mids-w207-final-project
This is the repo for the MIDS W207 Final Project.

# Setup Virtual Environment and Install Required Libraries
This project uses TensorFlow and Keras for Neural Network

## Add conda-forge channel
```shell
conda config --append channels anaconda
conda config --append channels conda-forge
```

## Install packages using the requirment files
### Linux
```shell
conda create --name <env_name> --file requiremen-linux.txt
```

### MacOS
```shell
conda create --name <env_name> --file requiremen-mac.txt
```

## Install packages manually
```shell 
conda create -n <env_name> python=3.9
```

```shell
conda install -y matplotlib
conda install -y pandas
conda install -y -c anaconda seaborn
conda install -y scikit-learn
conda install -y jupyter
conda install -y -c conda-forge ipykernel 


# Keras
conda install -y -c conda-forge keras
conda install -y -c conda-forge tensorflow
```

# Add the new virtual environment to Jupyter
If you want to use jupyter you could add the newly created environment to jupyter

```shell
conda activate <env_name>
conda install -y -c conda-forge ipykernel 
python -m ipykernel install --user --name <env_nam> --display-name "Python (<env_name>)"

```