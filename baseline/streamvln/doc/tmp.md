自己现在主要是在19服务器上安装环境

运行环境配置
按照github链接安装conda环境
conda create -n streamvln python=3.9
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines

clone streamvln的repo
git clone https://github.com/OpenRobotLab/StreamVLN.git
cd StreamVLN

安装其requirements
pip install -r requirement.txt
然后会遇到protobuf的问题，所以这时候需要将protobuf降级
pip uninstall protobuf -y
pip install protobuf==3.20.1

安装flash attention
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.4 --no-build-isolation
如果发现没有cuda tools
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
到此为止，用于evaluation的环境就算是已经ok了