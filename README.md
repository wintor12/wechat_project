## Quick Start
## Step 1: prepare the data
All data must be in the data folder. 
```bash
python prepare.py
```

## Step 2: preprocess
Chinese word segmentation, remove stopwords and noisy sentences, put everything to train/lda folder. 
```bash
python preprocess.py
```
Also need to copy y_.txt and v_.txt to train folder

## Step 3: run models
Regression:
```bash
python baseline.py --label LABEL --model MODEL
```
Classification:
```bash
python baseline.py --label LABEL --model MODEL --classification
```
