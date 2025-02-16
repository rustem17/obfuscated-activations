#!/bin/bash
# Remove any previously installed cupbearer package.
pip uninstall cupbearer -y

# Clean the pip cache.
pip cache purge

# Install main framework packages with pinned versions:
pip install torch==2.1.2
pip install transformers==4.45.2
pip install --upgrade huggingface_hub==0.25.2
pip install datasets==2.19.1

# Instead of installing transformer-lens from GitHub, install the released version.
pip install transformer-lens==2.7.0

# Install auxiliary packages with pinned versions:
pip install circuitsvis==1.43.2
pip install peft==0.13.1
pip install simple_parsing              # (version not specified)
pip install natsort==8.4.0
pip install scikit-learn==1.5.2
pip install matplotlib==3.9.2
pip install plotly                      # (version not specified)
pip install seaborn==0.13.2
pip install pandas==2.2.3
pip install wandb==0.18.3

# Install flash-attn at the required version.
pip install flash-attn==2.6.3 --no-build-isolation

pip install numpy==1.26.4

# The following packages did not have explicit version requirements in your list:
pip install pyod
pip install fire
pip install openai

# For cupbearer, we remove any previous version and then install the version from the provided archive.
pip install https://github.com/ejnnr/cupbearer/archive/abhay_update.zip
pip install torch==2.1.2
pip install torchvision==0.16.2

# (Optional commands which you can uncomment if needed:)
# sudo apt-get install python-tk python3-tk tk-dev
# huggingface-cli login
# wandb login
