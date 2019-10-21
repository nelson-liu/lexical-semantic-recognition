#!/usr/bin/env bash

set -e

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download


wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -f
conda update --yes conda
echo "Creating environment to run tests in."
conda create -q -n testenv --yes python="$PYTHON_VERSION"
cd ..
popd

# Activate the python environment we created.
echo "Activating Environment"
source activate testenv

# Install requirements via pip in our conda environment
echo "Installing requirements"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Install StanfordNLP models
python -c "import stanfordnlp; stanfordnlp.download('en', force=True)"
