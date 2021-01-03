FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.6.0-gpu-py3

#nothing out of ordinary. taking base image and uninstalling custom version of pytorch from aws and installing official cuda10 image. optionally : set up env proxy if training done in VPC mode
#custom torch version in this container have small conflicts with bert-base-uncased and probably other transformer models

RUN pip uninstall torch torchvision -y

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html