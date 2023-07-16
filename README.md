# ssorl
This repository contains the Pytorch implementation of [Semi-Supervised Offline Reinforcement Learning with Action-Free Trajectories](https://arxiv.org/abs/2210.06518)
by
[Qinqing Zheng](https://enosair.github.io/),
[Mikael Henaff](http://www.mikaelhenaff.com/),
[Brandon Amos](http://bamos.github.io/),
and [Aditya Grover](https://aditya-grover.github.io/).



If you use this code for your research, please cite us as:
```Bibtex
@inproceedings{zheng2023semi,
  title={Semi-supervised offline reinforcement learning with action-free trajectories},
  author={Zheng, Qinqing and Henaff, Mikael and Amos, Brandon and Grover, Aditya},
  booktitle={International Conference on Machine Learning},
  pages={42339--42362},
  year={2023},
  organization={PMLR}
}
```

## Requirements
Install the conda environment:
```console
conda env create -f conda_env.yml
conda activate ssorl
```

Update the path:
```console
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<your_conda_env_path>/lib
```

If you haven't installed patchelf, run:
```console
sudo apt-get install patchelf
```

## Example
Run the following command to train a `SS-TD3BC` agent for `hopper` with the `medium-v2` dataset, where 10%
trajecotories, whose returns are from the lower 50%,  contain actions.

```console
python main.py
```
This will produce the `exp-local` folder, where all the outputs are going to be logged including tensorboard blobs. One can attach a tensorboard to monitor training by running:
```console
tensorboard --logdir exp-local
```

## License
The majority of `ssorl` is licensed under CC-BY-NC, however portions of the project are available under separate license terms:
* D4RL dataset -  Creative Commons Attribution 4.0 License (CC-BY)
* D4RL code, transformers, Lamb - Apache 2.0 License
* stable-baselines3, Gym, decision-transformer - MIT License


