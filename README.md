# MLPScaling IA 2024 for La Quinta High School, La Quinta, CA, USA
Scaling MLP Models for Vision inspired by https://arxiv.org/abs/2306.13575 compute is provided by Shanghai AI Lab. Due to chinese restrictions Modelscope will be used initally and mirrored on huggingface after.

## Usage
For training and logging.
Intergrated
```bash
git clone https://github.com/BIRD-Laboratories/MLPScaling/
sh main.sh <MODELSCOPE API TOKEN>
```

## Outcomes/Process
This is testing: Scale, layer width ratios. These experiments will run for 4 days on 1xA100. 2 or 3 Days will be given to analysis operations.
Inital goals are to gather all checkpoints, gather all logs into one csv, visualize loss.
Strech goals are to make activation atlases, try evals on unseen data both inter and intra dataset.

## Context
My High School IB Math class requires an end of the 1st Semester statistics project. I decided to do this project due to it's simplicity but ease of scaling, filling my double mandate of impressive and doable. I luckily decided to not sign up for the IB Exam, so I am a little bit less limited in scope since I was worried the IB Graders would score low due to it's complex and difficult to explain nature. I thank both my chinese colaborators at Shanghai AI Lab et al along with my math teacher for guidance in this project. 

## Links
Paper: TBD
Modelscope: TBD, will be on https://modelscope.cn/models/puffy310/MLPScaling
Huggingface: TBD

## 11/20/24:
It is required to install mmengine from git
!pip install datasets modelscope transformers
!pip install git+https://github.com/open-mmlab/mmengine

## 11/22/24:
Training finally works for train_mlp_batches. Not yet tested at scale. Modelscope uploading is expected to work but unknown as of 18:29 PST. As of 21:26 testing at scale starts.

## 11/22.5/24:
23:27 PST
The training script has started, I made a seperate program to estimate the total time of all experiments.
Total time for all runs: 20181.81 minutes (336.36 hours, 14.02 days)
This might be an issue, my rough draft is due way before then. I might be able to ask Shanghai AI Lab for extra time but I cannot have that promise. I will most likely use Colab concurrently for the experiments after 3.5 days of experiments. Also how do you use newline in github??? I will have to consult both, not having the smaller datapoints isn't the end of the world but will still be an unfortunate outcome.

## 11/23/24
I've commitmaxxed to hell but another update. I ended up proceeding but i realized modelscope syncing was not enabled so i restarted the run. It seems like runtimes are limited to 12 hours and i need a way longer run time due to my currently unpredictable sleeping times. I editted the experiment creating to shuffle.
I forgot to state the time: 3:53PST.  If I am somehow still awake i'll see if modelscope uploading works properly.
