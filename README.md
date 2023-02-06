# D2IQA
Implementation of D2IQA described in [No-reference perceptual CT image quality assessment based on a self-supervised learning framework](https://iopscience.iop.org/article/10.1088/2632-2153/aca87d)


## Dataset

### In vivo clinical data with noise artifact (Mayo)
CT imagese from the [2016 Low Dose CT Grand Challenge dataset](www.aapm.org/GrandChallenge/LowDoseCT/)\\
Train/validation dataset consists of 7 patients.\\
The radiation dose corresponding to each folder is as follows:
|Folder Name|full|full_0.5|quarter|quarter_0.5|quarter_1.0|quarter_1.5|quarter_2.0|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Radiation Dose|100%|50%|25%|20%|12%|7%|5%| 

### Anthropomorphic phantom data with real noise (Phantom)
The radiation dose corresponding to each folder is as follows:
|Folder Name|level1_100|level2_050|level3_025|level4_010|level5_005|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Radiation Dose|100%|50%|25%|10%|5%| 

### In vivo clinical data with noise artifact (CQ500)
CT imagese described in [Development and Validation of Deep Learning Algorithms for Detection of Critical Findings in Head CT Scans](https://arxiv.org/abs/1803.05854) available at [qure.ai](http://headctstudy.qure.ai/#about)\\
The noise level corresponding to each folder is as follows. Note that the folder names have nothing to do with radiation dose.
|Folder Name|quarter|quarter_0.5|quarter_1.0|quarter_1.5|quarter_2.0|quarter_2.5|quarter_3.0|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Noise Level|1|2|3|4|5|6|7| 
