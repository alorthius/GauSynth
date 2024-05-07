<h1 align="center">GauSynth: </h1>
<h2 align="center" style="position: relative; top: -30px;">Diffusion-based Reimagination for 3D Object Synthesis with Gaussian Splatting </h2>

<p align="center" style="position: relative; top: -30px;"><a href="https://alorthius.github.io/GauSynth/"> Project Page </a></p>



**GauSynth** is a 3D reimagination pipeline composed of the recent 2D generative model [SD-XL](https://arxiv.org/abs/2307.01952), geometry reconstruction based on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), and Super-Resolution upscaling model [Swin2SR](https://arxiv.org/abs/2209.11345) to address the low-quality issue of the existing 3D generative models. The proposed system generalizes to various inputs, such as renderings of _digital 3D assets_ or turn-around videos of a _real-life object_.

https://github.com/alorthius/GauSynth/assets/73172589/92ed61cf-c108-40d5-9aeb-6b74d1aa5c51

https://github.com/alorthius/GauSynth/assets/73172589/bc82ac80-ed69-4ae9-be0b-839806e4fb19

https://github.com/alorthius/GauSynth/assets/73172589/11a07119-f0d3-4d0e-aa69-338d76024401

---

<h2 align="center">Requirements</h2>

**Hardware**:
* CUDA-ready GPU
* GPU with at least 8 GB of VRAM (tested on 12 GB of VRAM)
* At least 32 GB of RAM (tested on 32 GB of RAM + 40 GB of swap space)
* At least 20 GB of storage (for several SD-XL checkpoints and verbose intermediate results)

**Software**:
* Python 3.11
* Virtual venv / conda environment
* Linux-based OS (tested on Ubuntu 22.04)
* CUDA SDK 11/12 (tested on 12.1)
* GCC and G++ compilers

---

<h2 align="center">Installation</h2>

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
We use Fooocus for the implementation of image-to-image inference of Stable Diffusion with the ControlNet conditioning models.
```shell
python -m srcipts.update_focus_model
cd Fooocus/
pip install -r requirements_versions.txt
wget "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors?download=true" -O models/checkpoints/juggernautXL_v9Rundiffusion.safetensors
python entry_with_update.py  # after all downloads are finished and ui is launched, terminate it

cd ..  # back to repo root
```

### Fooocus-API
We leverage the convenient API calls to the Fooocus system via Fooocus-API.
```shell
pip install fastapi==0.103.1 pydantic==2.4.2 pydantic_core==2.10.1 python-multipart==0.0.6 uvicorn[standard]==0.23.2 colorlog requests sqlalchemy packaging rich chardet
```

Start server, **do not shutdown** and leave the process running:
```shell
cd Fooocus-API/
# for low VRAM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --always-offload-from-vram
# for large VRAM
python main.py
```

### Ezsynth
Compile Ezsynth, a community open-source implementation of Ebsynth on top of the original repo, which we use as an interpolation model.
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
For running Structure from Motion pipeline, we use Colmap. To launch it on GPU, compile it from source:
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
3D Gaussian Splatting is used for the final dense geometry reconstruction. The installation instruction is taken from the source repository.
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
Finally, we can launch the GauSynth demo! Do not forget to launch the Fooocus-API service as we described above. The main demo is started from the project root as:
```shell
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python demo.py
```

![](./docs/static/images/gui.png)

---
<h2 align="center">Usage</h2>

The GUI demo is divided onto separate stages for showing each intermediate results. Most of the previews displayed feature the videos merged from processed individual images. For manual inspection and reuse of the assets, all of the intermediate processes frames are saved locally automatically after completing a certain step.

The usage flow is intended for user to complete them from top to bottom, moving from left to right in the components. We further review each of the major stage in the order intended for usage.


### Preprocessing
Thus, firstly, the **Directory** name should be stated in the top-left Preprocessing corner. To split the input video into separate frames, the **Original fps** should be stated. Additionally, user can control the taken **Number of frames** to filter the long-lasting video to much smaller amount of images.

### Model Sheet
To create a model sheet, only several key frames should be used from the all frames collection. The user should select its **Dimensionality** value N to create the respective grid of size NxN composed of key frames. They are selected to be evenly-spaced and capture the whole turn-around content.

### Reimagination
It features SD-based block for reimagining key frames only. The basic options consist of **Text prompt** field, **Denoising Strength** parameter for stating how much changes should be introduced to the original, and the **Seed** value, which can be either freezed to some integer or randomly sampled by writing _-1_. There are 2 SD-XL checkpoints options to choose from.

The ControlNet guidance of the generation consists of 2 preprocessors: PyraCanny Edges and Image-Prompt Adapter. Each of them have the respective **Weight** and **Stop at** parameters. Weight controls the amount of guidance it introduces, which the greater values meaning the greater influence. The Stop at criterion states after which step the guidance should be removed (it is recommended to eliminate it from the final stages for better refinement).

Also, PyraCanny edge detection features **Low** and **High** thresholds to be tuned for better generation quality. The estimated edge map can be previewed _before_ running the generation for better parameters selection.

The **Reimagine** button launches the reimagination of the previously created model sheet.

### Interpolation
This block is used to interpolate frames from key ones to achieve the reimagination of all the intermediate images. After clicking the **Interpolate frames** step, we should have reimagined all the frames from the input processed video.

We also post-process the described interpolation to remove the background and eliminate some produced artifacts by running the **Blend** process.

### Super Resolution
We refine and enhance the details number of the reimagined frames by running **Upscale** stage.

### SfM
The COLMAP SfM pipeline is called here to reconstruct a sparce point cloud. We display some important estimation information, such as number of images, which we want to be close to the input _Number of frames_ of the original video from the Preprocessing stage.

### 3D Gaussian Splatting
We provide 2 options to run the reconstruction: on all the reimagined _or_ original frames. By running both, we can compare the metrics and the visual quality of the reimagined and original renderings. The user can toggle the **Training iterations** parameter for the final number of optimization steps, still the intermediate evaluation will be done on several iterations.

### Local saves structure

All the intermediate images and models will be saved in the `demo_outputs_dir/{Directory field}` folder locally. We shortly describe its components saved after running the respective stages:
```
demo_outputs_dir
|_____ {DIRECTORY} (from GUI field)
           |_____ filtered_frames_colmap  - frames passed for colmap reconstruction
           |_____ filtered_frames_sd      - frames passed for 2D generive models
           
           |_____ orig_sheets             - original model sheets
           |_____ reimagine_sheets        - reimagined model sheets

           |_____ ebsynth_all             - interpolated frames after SD reimagination
           |_____ ebsynth_transparent     - interpolated frames with no background
           |_____ orig_transparent        - original frames with no background
           |_____ blend_transparent       - post-processed interpolated images with no background
           |_____ sr_frames               - frames after Super Resolution enhancement
           
           |_____ colmap                  - directory with the estimated point-cloud model
                    |_____ distorted
                    |_____ images
                    |_____ input
                    |_____ sparse
                    |_____ stereo

           |_____ gs                      - directory with the Gaussian Splatting reconstructions
                   |_____ reim_{TRAIN_ITERS}-{DATE-TIME}          - reconstruction of reimagined frames
                           |_____ point_cloud
                           |_____ renders_{TRAIN_ITERS}
                           |_____ test
                           |_____ train
                   |_____ orig_{TRAIN_ITERS}-{DATE-TIME}          - reconstruction of otiginal frames
                           |_____ ... same as in reim ...
           
           |_____ new_videos              - folder with the preview videos displayed in the demo
```

---

<h2 align="center">Acknowledgement</h2>

GauSynth is built upon the following awesome open-source projects:
* [Fooocus](https://github.com/lllyasviel/Fooocus)
* [Fooocus-API](https://github.com/mrhan1993/Fooocus-API)
* [Ezsynth](https://github.com/Trentonom0r3/Ezsynth)
* [Swin2SR](https://github.com/mv-lab/swin2sr)
* [Colmap](https://github.com/colmap/colmap)
* [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

---

<h3 align="center">Contributors</h3>

* [Anna-Alina Bondarets](https://github.com/alorthius)
* Taras Rumezhak
