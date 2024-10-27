# No-reference perceptual CT image quality assessment based on a self-supervised learning framework

This repo contains the supported code and configuration files to reproduce image quality assessment results of [No-reference perceptual CT image quality assessment based on a self-supervised learning framework](doi.org/10.1088/2632-2153/aca87d). 

## Updates

***10/26/2024*** Initial commits

## Usage

### Inference

### Generating data for model testing

To test D2IQA, prepare images with various noise levels in the following structure:
```
data_root/
└── nimg/
    ├── L506/
    │   ├── full/
    |   |   ├── 120.0.tiff
    │   │   └── 121.0.tiff
    │   ├── ...
    │   ├── quarter_2.0/
    │   │   └── ...
    └── L067/
        ├── full/
        │   └── ...
        └── ...
```

Once the dataset is ready, uncomment the functions related to generating test sets.

- To generate a test set per patient, uncomment `make_ptestset` or `make_ptestset_sep`. `make_ptestset` inserts objects with random configuration combinations, while `make_ptestset_sep` inserts objects for every possible combination of contrast and size, saving each in individual folders.

- To generate a test set per slice image, uncomment `make_ptest_all` or `make_ptest_sep`. Similarly, `make_ptest_all` inserts objects with random configurations, and `make_ptest_sep` inserts all possible combinations of size and contrast configurations in individual folders. Modify the fid list in the `make_ptest_all` or `make_ptest_sep` functions as needed.
```
python make_data.py --dataset new_folder_name --root /path/to/data_root/ --mc True --r 7 8 9 10 --c 0.1 0.09 0.08 --pid patient_id
```

**Notes**:

- Make sure to end the `root` argument path with a '/'

### Calculating IQA scores using D2IQA

Install mmdetection:
```
pip3 install torch==1.10.1 torchvision==0.11.2 --index-url https://download.pytorch.org/whl/cu113
pip install -U openmim==0.1.5
mim install mmcv-full==1.4.1
```

Download the model from this [link](https://drive.google.com/file/d/1Ca6jENWOcyFSu9NehDPY27646lDHpzZQ/view?usp=drive_link) and place the downloaded folder in `./work_dirs`. Then, uncomment the code in `test.py` according to your data structure and run: 
```
python test.py --dataset test_data_folder_name --root /path/to/data_root/ --test_folder test_folder_name --date 0805_5
```

**Notes**:

- If you are using classes other than the four classes `triangle`, `circle`, `square`, and `star`, update the CLASSES variable accordingly in `mmdet/datasets/coco.py`.

## Training

### Generating data for model training

To train D2IQA for CT abdomen images, download the dicom images from the [2016 Low Dose CT Grand Challenge dataset](www.aapm.org/GrandChallenge/LowDoseCT/) and preprocess them. In this project, we normalized images to a range of 0 to 1, using a window width of 1400 and level of -300. The preprocessed dataset should have the following data structure.
```
data_root/
└── Mayo-Clinic/
    ├── tiff/
    │   ├── raw/
    │   ├── processed/
    │   │   └── dataset.csv
    ├── models/
    │   ├── model_weights.pth
    ├── scripts/
    │   └── preprocess.py
    └── tiff
        └── full_1mm
        │   ├── L067
        |   |   ├── 000.tiff
        |   |   ├── 001.tiff
        |   |   └── ... 
        │   ├── L096
        |   |   ├── 000.tiff
        |   |   ├── 001.tiff
        |   |   └── ...
        │   └── ...
        └── quarter_1mm
            ├── L067
            |   └── ...
            ├── L096
            |   └── ...
            └── ...

```

Once the dataset is ready, uncomment `make_trainset` or `make_full_trainset`, and then run:
```
python make_data.py --dataset new_folder_name --root /path/to/data_root --mc True --r 7 8 9 10 --c 0.1 0.09 0.08 
```

**Notes**:

- `make_trainset` uses both full and quarter dose images, while `make_full_trainset` uses only full dose images.
- Do not include the words `train`, `val`, or `test` in the new folder name.
- If `mc` is `False`, the inserted lesions will be circles only.
- Modify the radius (`r`) and contrast (`c`) as needed.

Then, run the command below to split the data into training and validation sets.
```
python data_split_pid.py --dataset new_folder_name --root /path/to/data_root
```

Lastly, create json files by running `mayo2coco.py`.

```
python mayo2coco.py --dataset new_folder_name --root /path/to/data_root
```

### Train the model using mmdetection

Train the detector using [mmdetection](https://github.com/open-mmlab/mmdetection). D2IQA uses Cascade R-CNN with ResNet-50 backbone. The configuration file for training D2IQA is availabel at this link. Update `data_root` to point where your dataset folder and `work_dirs` to the directory where you want to save the model.

## Citing D2IQA
```
@article{lee2022no,
  title={No-reference perceptual CT image quality assessment based on a self-supervised learning framework},
  author={Lee, Wonkyeong and Cho, Eunbyeol and Kim, Wonjin and Choi, Hyebin and Beck, Kyongmin Sarah and Yoon, Hyun Jung and Baek, Jongduk and Choi, Jang-Hwan},
  journal={Machine Learning: Science and Technology},
  volume={3},
  number={4},
  pages={045033},
  year={2022},
  publisher={IOP Publishing}
}
```
