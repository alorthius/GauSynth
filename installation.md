### Requirements
**Hardware**:
* CUDA-ready GPU
* GPU with 8 GB of VRAM (tested on 12 GM of VRAM)
* At least 32 GB of RAM (tested on 32 GB of RAM + 40 GB of swap space)
* At least 30 GB of storage (for several SD checkpoints)

**Software**:
* Python 3.11
* Virtual venv / conda environment
* Linux-based OS (tested on Ubuntu 22.04)
* CUDA SDK 11/12 (tested on 12.1)
* GCC and G++ compilers

This project depends on many libraries required to be compiled from source. We list all the instructions we used for our particular setup, which may differ for your machine. For more detailed installation please refer to the source repositories (submodules of this repo).

### Clone submodules
```shell
git clone https://github.com/alorthius/3D-diffusion-splatting
git submodule update --init --recursive
```

### Basic dependencies
```shell
# cuda 12.1
pip install torch torchvision
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# cuda 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xformers --index-url https://download.pytorch.org/whl/cu118

pip install torchmetrics[image]
pip install rembg  # background remover

sudo apt-get install ffmpeg  # video processing
```

### Fooocus
```shell
python -m srcipts.update_focus_model
cd Fooocus/
pip install -r requirements_versions.txt
wget "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors?download=true" -O models/checkpoints/juggernautXL_v9Rundiffusion.safetensors
python entry_with_update.py  # after all downloads are finished and ui is launched, terminate it

cd ..  # back to repo root
```

### Fooocus-API
```shell
pip install fastapi==0.103.1 pydantic==2.4.2 pydantic_core==2.10.1 python-multipart==0.0.6 uvicorn[standard]==0.23.2 colorlog requests sqlalchemy packaging rich chardet

cd Fooocus-API/
```

Start server, do not shutdown and leave the process running:
```shell
# for low VRAM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --always-offload-from-vram
# for large VRAM
python main.py
```

### Ebsynth
Compile Ezsynth, a community open-source implementation of Ebsynth on top of original repo
```shell
wget "https://drive.google.com/uc?export=download&id=1fubTHIa_b2C8HqfbPtKXwoRd9QsYxRL6" -O raft-sintel.pth
cp raft-sintel.pth your_python_env/lib/python3.11/site-packages/ezsynth/utils/flow_utils/models/

pip install phycv

cd Ezsynth/ebsynth
./build-linux-cpu+cuda.sh  # compile Ebsynth

# IMPORTANT: change 62'th line in Ezsynth/ezsynth/EZMain.py to "self.save_results(self.output_folder, f"{str(i).zfill(2)}.png", self.results[i])"

cd ../..  # back to repo root
```

### Colmap
For running Colmap on GPU, compile it from source (repo already cloned here):
```shell
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# colmap compilation fails on newer gcc/g++ versions
sudo apt-get install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

cd colmap
mkdir build
cd build
cmake -GNinja -D CMAKE_CUDA_ARCHITECTURES=86 ..
ninja
sudo ninja install

pip install open3d  # for colmap visualizations

cd ../..  # back to repo root
```

### 3D Gaussian Splatting
```shell
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev

cd gaussian-splatting
pip install plyfile
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/

cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -GNinja
cmake --build build -j24 --target install

cd ../..  # back to repo root
```

### Launch demo
```shell
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python demo.py
```