# Multi-Objective Evolutionary GAN
Official Multi-Objective Evolutionary GAN implementation.
![Diagram](imgs/MO-EGAN.png?raw=true "Diagram")

# installation

This framework is developed for this PC's configuration:

- Ubuntu 18.04 LTS or 20.04 LTS
- Python 3 or pypy

You can installd it, in CPU mode or GPU one.

N.B. Only the nVidia GPU are suppoted.

## CPU

For python3, only cpu:

        apt install python3-dev
        python3 -m pip install -r requirements.txt


## GPU
Instead, for gpu version, you have to install:

        apt install cmake nvidia-cuda-toolkit

then you have to manually install:

[cuDNN 7.6](https://developer.nvidia.com/cudnn)

[libgpuarray 7.2 (pygpu)](http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install-system-library-as-admin) (git checkout v7.2)

Now, you can install the python gpu dependencies:

        apt install python3-dev
        pypy -m pip install -r requirements_gpu.txt

Or the pypy dependencies:

        apt install pypy3-dev libfreetype-dev libopenblas64-dev
        pypy -m pip install -r requirements_gpu.txt
        

# Citation
If you use this code for your research, please cite [our paper](https://dl.acm.org/doi/abs/10.1145/3377929.3398138?casa_token=hJYwpbrljEgAAAAA:s9ycwBANLA6ReFbpx8Ecyd-S6zhwTUIEoejoswoW3CtUYeOWRDK57cVUXW9GNE0W8mPvVV8NvbWy).

    @inproceedings{baioletti2020multi,
    title={Multi-objective evolutionary GAN},
    author={Baioletti, Marco and Coello, Carlos Artemio Coello and Di Bari, Gabriele and Poggioni, Valentina},
    booktitle={Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion},
    pages={1824--1831},
    year={2020}
    }
