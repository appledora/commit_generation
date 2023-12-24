# Automatic Generation of Git Commit messages from Changes in Code
This repository contains the code and outputs for the `CS505: Natual Language Processing` course project. The objective of this work is to explore the performance of different machine learning models in generating commit messages from changes in code.

# Requirements
```
pip install nltk==3.6.2 scipy==1.5.2 pandas==1.1.3 krippendorff==0.4.0 scikit-learn==0.24.1 sumeval==0.2.2 sacrebleu==1.5.1
pip install sentencepiece==0.1.95 transformers==4.8.2
```

## Dataset
The dataset used for this project is the [NNGen Dataset]() which contains  .... 

## Repository Structure
The repository is structured as follows:
```
.
├── baseline
│   ├── baseline.ipynb
│   ├── bleu.perl
│   └── prediction.msg.txt
├── EDA.ipynb
|── evaluators
│   ├── bleu.py
|   └── rouge.py
├── README.md
├── seq2seq
│   └── Seq2Seq.ipynb
└── transfer_learning
    ├── beam_algo.py
    ├── GCG_model.py
    ├── prediction.msg.txt
    └── train_CBert.py

```

- `EDA.ipynb`: Contains the code for the Exploratory Data Analysis of the dataset.
- `baseline`: Contains the code for the baseline model. It also contains the `prediction.msg.txt` file which contains the predicted commit messages.
- `seq2seq`: Contains the code for the seq2seq model and corresponding outputs.
- `transfer_learning`: Contains the code for the transfer learning model, helper scripts and corresponding outputs.
- `evaluators`: Contains the code for the evaluation metrics.

## Results
The results of the project are summarized in the following table:

|                          | **BLEU-4** | **Rouge-L** |
|--------------------------|------------|-------------|
| **Baseline**       | 16.41      | 26.93       |
| **Seq2Seq**              |   15         |         15    |
| **Transfer Learning**    | 15.08      | 18.23         |
| **Zero-shot Inference** | 7.5 | 13.67 |


## Authors
- [Nazia Tasnim](appledora.github.io)
- [Naima Abrar]()
