# TeamTHARA
Codebase for the project 'Detection of Persuasion Techniques in Memes' 

## Downloading the datasets
To access the PTC Corpus and SemEval2024 datasets, please register here:

* [SemEval2024 dataset](https://propaganda.math.unipd.it/semeval2024task4/) 
* [PTC Corpus](https://propaganda.math.unipd.it/ptc/)

The following file structure should be reproduced when adding the datasets (and other files) to the project directory:

```
memes_project
├── image_data
│   ├── dev_images
│   │   └── prop_meme_41.png
│   │   .....
│   ├── train_images
│   ├── validation_images
│
├── ptc_corpus
│   ├── ptc_datasets
│   │   ├── train-articles
│   │   ├── dev-articles
│   │   └── train-task-flc-tc.labels
|   |   └── dev-task-flc-tc.labels
│
└── scorer-baseline
│   └── subtask_1_2a.py
│
└── semeval2024_dev_release
    ├── dev_gold_labels
    │   └── dev_subtask1_en.json
    │   └── dev_subtask2a_en.json
    │
    ├── subtask1
    │   └── dev_unlabeled.json
    │   └── train.json
    │   └── validation.json
    │
    └── subtask2a
        └── dev_unlabeled.json
        └── train.json
        └── validation.json
```

## Preprocessing the PTC Corpus
In order to use the PTC Corpus for training, run the following command to preprocess the data and save to CSV files:

```
python ptc_corpus/preprocessing.py
```

## Pretraining with PTC Corpus
To pretrain the text encoders with the PTC Corpus, run the following:

```
python ptc_corpus/train.py
```

Include the path to the checkpoint in task1_train.py or task2_train.py.

## Training
Run the following to train the model.

**For subtask 1**:
```
python task1/task1_train.py
```


**For subtask 2**:
```
python task2/task2_train.py
```

## Note
The proposed architecture is designed to use RoBERTa and ViT. If using any other models, please follow the comments in the following files to make the necessary changes:

* **Subtask 1**: task1_train.py, task1_collate.py
* **Subtask 1**: task2_train.py, task2_collate.py, task2_dataset.py
