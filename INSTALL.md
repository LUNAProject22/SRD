## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- PyTorch >= 1.2
- torchvision >= 0.4
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- pandas

### Working virtual environment and platform
- Python 3.8
- PyTorch 1.8.0
- torchvision 0.9.0
- CUDA toolkit 11.1
- GPU: NVIDIA GeForce GTX 1080 Ti
- Driver Version: 470.103.01
- CUDA Version: 11.4

### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name sgg python=3.8
conda activate sgg

# this installs the right pip and dependencies for the fresh python
conda install ipython scipy h5py pandas scikit-learn

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python tensorboard overrides
pip install git+https://github.com/openai/CLIP.git

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
# better to checkout 22.04-dev branch to avoid this issue
# https://github.com/NVIDIA/apex/issues/1532
git fetch origin 22.04-dev:22.04-dev
git checkout 22.04-dev
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/LUNAProject22/SRD.git
cd SRD

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR
