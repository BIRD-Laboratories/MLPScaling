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
```bash
!pip install datasets modelscope transformers
!pip install git+https://github.com/open-mmlab/mmengine
```

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
5:38PST: I had to restart again because I forgot to add intermediate checkpointing. I might not end up using these datapoints now but during the Neurips Expansion it will help.

## 11/23.5/24
14:22 PST, last night at approx 5:00 i turned off my experiment. I'll go turn it on again now. i'll start working on eval code shortly. I had to edit modelscope functionality so I can offload all models properly. 15:32 PST, I am still having issues with syncing on modelscope, hopefully they are nearing a proper fix but I am starting to get slighlty frusterated. I am in much better standing than the day before but the battle is by no means over. 17:12 I have gotten the folder functionality to work I am so happy. The code should finally run uninterupted. I'll take a break for a bit and then I need to work on an eval harness. As of 22:30 I still have not gotten the code fully working. I am ahead of schedule still. The modelscope syncing will be done via a branch system rather than folders all on one branch. I updated the experiment profile to have 8 big experiments at the start and then do more ratio experiments to ensure I get a good data mix. Other than that I am pretty tired and nearing a frusteration level that will hold back my work so I am going to sleep earlier today.

## 11/24/24
I was originally going to take a longer break but no. It's 11:39 PST at the moment, I just woke up. I turned off my run last night due to another error. I am really hoping I can finally start the full run already. 
15:26 I am going to add color and finally solve the syncing issue.

## 11/24.5/24
I have added color into the code again, I finally figured out how to fix the RGB issue. Other than that I have also switcheh the syncing to be in the SH script, NOT the python script. I am hoping this shift will make the code work the way I want it to.

## 11/25/24
I am just going to make the modelscope syncing optional and not use it. i need to get started already. 1:29 PST is the time. i wrote this down mow but i must sleep. Modelscope syncing can be done in the expanded version. 

## 11/25.5/24
I have gotten the training script to run. I am deciding to leave the syncing issues for later and my run has finally start, about on time to be on schedule for everything else. I used tmux so that I can exit the SSH without any issues. I will be starting a prelim report and hopefully finishing it soon. 20:08 it seems training doesn't really work (?) I might have set the learning late too low.

## 11/26/24
I got experimental data in and it is on my local drive. For now I am taking a break.

## 11/26.5/24
23:14 PST Within the script i'll add an option to start at a certain line in the csv. This is so I can resume training. I'll have it be based on the last experiment(arguement) and continue the one after. 23:53 PST I spend almost an hour trying to get the idea to work just for it not to work, will use some of the expanded experiments and then just cut the collecting short.

## 11/29/24
15:49 PST I have taken an extremely long break. near the start of the readme I will add some documentation and methodology. 
