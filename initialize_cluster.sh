#!/bin/bash
# License: Public Domain.

#########################################################
# System-wide packages
#########################################################
# Interrupts the script in case of failures and puts the cluster on ERROR
set -e

# Update metadata before installing anything
apt-get update
# Installing the required packages
apt-get install -y libblas-dev liblapack-dev gfortran make wget curl
apt-get install -y libpng-dev libfreetype6-dev libxft-dev pkg-config
apt-get install -y build-essential python3-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev
apt-get install -y libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils
#########################################################
# Ends
#########################################################

# This part imports the code for "Python Setup" and exposes a couple of environment variables
# - $PIP_EXECUTABLE and $PYTHON_EXECUTABLE

#########################################################
# Python 3.5.2 Setup
#########################################################

# Backing up location of the home directory
export OLD_HOME=$HOME

# Changing home directory for Miniconda installation
# The use of /opt for add-on software is a well-established practice in the UNIX community.
# Also this makes miniconda available in the same location for all users
export HOME=/opt
cd $HOME
# Miniconda version for Python 3.5.2
wget https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh
/bin/bash Miniconda3-4.2.12-Linux-x86_64.sh -b

# Defining Python related paths
export PYTHON_BIN_PATH=$HOME/miniconda3/bin
export PYTHON_EXECUTABLE=$PYTHON_BIN_PATH/python
export PIP_EXECUTABLE=$PYTHON_BIN_PATH/pip

# Setting (from the backed up location) the actual home directory
export HOME=$OLD_HOME

# Telling Spark to use the newly installed Python version
echo "export PYSPARK_PYTHON=$PYTHON_EXECUTABLE" | tee -a  /etc/profile.d/spark_config.sh \
    /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "export PYTHONHASHSEED=0" | tee -a /etc/profile.d/spark_config.sh /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf

# Upgrading pip
$PIP_EXECUTABLE install --upgrade pip

#########################################################
# Ends
#########################################################


# Make a google cloud json settings file so that machines in the cluster can identify what project they are in
mkdir /usr/local/etc/etl
echo '{"project_id":"coolblue-bi-platform-prod"}' >> /usr/local/etc/etl/google_cloud_key.json

# TODO: Find a nicer way to specify these dependencies automatically from gfm/data-science-utilities/cloud-forecasts
# This fixes the _NamespacePath error for google-cloud libraries installation on Python 3.5
$PIP_EXECUTABLE install setuptools==39.0.1 --ignore-installed
$PIP_EXECUTABLE install requests==2.18.4
$PIP_EXECUTABLE install h5py==2.6.0
$PIP_EXECUTABLE install py4j==0.10.6
$PIP_EXECUTABLE install configobj==5.0.6
$PIP_EXECUTABLE install datadog>=0.15.0
$PIP_EXECUTABLE install diskcache>=2.4.1
$PIP_EXECUTABLE install flask>=0.11.1
$PIP_EXECUTABLE install gcloud==0.18.3
$PIP_EXECUTABLE install google-api-python-client==1.6.4
$PIP_EXECUTABLE install google-cloud-core==0.24.0
$PIP_EXECUTABLE install google-cloud-bigquery==0.24.0
$PIP_EXECUTABLE install google-cloud-storage==1.0.0
$PIP_EXECUTABLE install google-cloud-datastore==1.0.0
$PIP_EXECUTABLE install jsonschema==2.5.1
$PIP_EXECUTABLE install matplotlib>=1.5.3 # For glmnet
$PIP_EXECUTABLE install numpy==1.11.3
$PIP_EXECUTABLE install openpyxl>=2.3.2
$PIP_EXECUTABLE install pandas==0.19.2
$PIP_EXECUTABLE install paramiko==2.0.2
$PIP_EXECUTABLE install PyYAML==3.11
$PIP_EXECUTABLE install scikit-learn==0.18.1
$PIP_EXECUTABLE install scipy==0.17.1
$PIP_EXECUTABLE install shap==0.24.0
$PIP_EXECUTABLE install statsmodels>=0.6.1
$PIP_EXECUTABLE install sortedcontainers==1.5.7
$PIP_EXECUTABLE install workalendar==0.8.1
$PIP_EXECUTABLE install xgboost==0.80
$PIP_EXECUTABLE install typing==3.6.4
$PIP_EXECUTABLE install lightgbm==2.2.2 # --install-option=--gpu
$PIP_EXECUTABLE install tables==3.4.4
# GLMnet:
# $PIP_EXECUTABLE install -e git+https://github.com/bbalasub1/glmnet_python.git@97cb8400a2e5e6c2509d3a3448a78c0c37f6c734#egg=glmnet_python
