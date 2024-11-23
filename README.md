# MLPScaling IA 2024 for La Quinta High School, La Quinta, CA, USA
Scaling MLP Models for Vision inspired by https://arxiv.org/abs/2306.13575 compute is provided by Shanghai AI Lab. Due to chinese restrictions Modelscope will be used initally and mirrored on huggingface after.
This is testing: Scale, layer width ratios. These experiments will run for 4 days on 1xA100. 2 or 3 Days will be given to analysis operations. 
Inital goals are to gather all checkpoints, gather all logs into one csv, visualize loss.
Strech goals are to make activation atlases, try evals on unseen data both inter and intra dataset.

## 11/20/24:
It is required to install mmengine from git
!pip install datasets modelscope transformers
!pip install git+https://github.com/open-mmlab/mmengine

## 11/22/24:
Training finally works for train_mlp_batches. Not yet tested at scale. Modelscope uploading is expected to work but unknown as of 18:29 PST. As of 21:26 testing at scale starts.
