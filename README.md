# SSML-FD
Unsupervised Vehicle Re-Identification based on Self-supervised Metric Learning using Feature Dictionary

## Abastract

The key challenge of unsupervised vehicle re-identification (Re-ID) is learning discriminative features from unlabelled vehicle images. Numerous methods using domain adaptation have achieved outstanding performance, but those methods still need a labelled dataset as a source domain. This paper addresses unsupervised vehicle Re-ID, which no need any types of a labelled dataset, though a Self-supervised Metric Learning (SSML) based on a feature dictionary. Our method initially extracts features from vehicle images and stores them in a dictionary. Thereafter, based on the dictionary, the proposed method conducts dictionary-based positive label mining (DPLM) to search for positive labels. Pair-wise similarity, relative-rank consistency, and adjacent feature distribution similarity are considered to find images that may belong to the same vehicle to a given probe image. The results of DPLM are applied to dictionary-based triplet loss (DTL) to improve the discriminativeness of learnt features and refine the quality of the results of DPLM progressively. The iterative process with DPLM and DTL boosts the performance of unsupervised vehicle Re-ID. Experimental results demonstrate the effectiveness of the proposed method by producing promising vehicle Re-ID performance without a pre-labelled dataset.



## Dependencies

This project mainly complied with Python3.6, Pytorch 1.3. All details are included in the 'requirement.txt'

~~~
#Setting the environment
pip install -r requirements.txt
~~~


## File configuration

<br>
├── data #Extract dataset to this directory. <br>
├── experiments <br>
├── lib <br>
├── logs <br>
├── models <br>
│   └── imagenet #Extract backbone network checkpoint here <br>
├── output # Extract the checkpoints to reproduct the results. <br>
└── tools <br>



## Dataset preparation
[Veri-776](https://vehiclereid.github.io/VeRi/) dataset and [Veri-wild](https://github.com/PKU-IMRE/VERI-Wild) dataset are used for this work.
To train the proposed method, change the ditectory names to 'bounding_box_train' (training set), 'bounding_box_test' (test set), 'query' (query set).
In using Veri-Wild dataset, to evaluate the small, medium, and large test set. you have to make the directories as follows:
~~~
output_test_middle_img_path =  './test_middle/'
output_query_middle_img_path = './query_middle/'

output_test_small_img_path =  './test_small/'
output_query_small_img_path = './query_small/'
~~~

Please refer 'preprocessiong_dataset/veri_wild_transform.py' file to conduct experiments for veri-wild dataset.



## Backbone network (ResNet-50) Reference
You can download the backbone network model from [here](https://drive.google.com/file/d/1rfCcrOzIWNWakA3BYkqp5om2_nI5Ftr8/view?usp=sharing). Save the weight file on './models/imagenet'




## How to train and test
~~~
./do_exp.sh
~~~



## Reproduce the experimental results

You can download the checkpoint files to reproduct the experiment results from [here](https://drive.google.com/drive/folders/1iglDV_H1obl5vopL6pFA6KiY7s-8fb0S?usp=sharing). After download it. Extract the file under the './outputs/veri776' or './outputs/veri-wild' depending on what you want to reproduct.



## Code reference.
* The code is mainly encouraged by [GSMLP-SMLC](https://github.com/andreYoo/GSMLP-SMLC.git) and [MLCReID](https://github.com/kennethwdk/MLCReID)



## Current issue[!!!].
Since the scale of Veri-Wild dataset is too large, we may have a segment fault issue when you run the training code for the dataset. We provide a source code file to train the proposed method using CPU and DRAM settings. 'train_with_cpu.py' is it. Unfortunately, Training our model based on CPU is extremely slower than GPU-based learning. It may need over than 24 hours for one epoch.

