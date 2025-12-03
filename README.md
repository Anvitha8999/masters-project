# Real-Time Emotion Recognition from Conversations

A deep learning system for real-time emotion recognition from text and audio modalities using CNN-BiLSTM architecture with attention mechanism.

## Project Overview

This system recognizes 7 emotions from conversational data:
**Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise **

**Key Features:**

- Multimodal analysis (text + audio)
- CNN-BiLSTM with attention mechanism
- Real-time predictions (< 200ms)
- Trained on MELD dataset

## Project Structure

```
anvitha-project/
├── configs/
│   └── config.yaml              # Training configuration
├── data/
│   └── meld/
│       ├── text/                # CSV files with utterances
│       ├── video/               # Video files (.mp4)
│       └── audio/               # Audio files (.wav)
├── scripts/
│   ├── clean_dataset.py         # Clean CSV files
│   ├── extract_audio.py         # Extract audio from videos
│   ├── train.py                 # Train the model
│   └── inference.py             # Run predictions
├── src/
│   ├── model.py                 # Model architecture
│   ├── data_preprocessing.py    # Preprocessing functions
│   └── dataset.py               # PyTorch Dataset
├── models/                      # Saved model checkpoints
├── logs/                        # Training logs
├── requirements.txt             # Dependencies
└── README.md                    # This file
```


**3.2 Extract and Organize Files**

After downloading, organize the files in your project:

**Step 1:** Create the directory structure:

```powershell
cd anvitha-project
mkdir data\meld\text
mkdir data\meld\video
mkdir data\meld\audio
```

**Step 2:** Extract video files:

- Extract `train.tar.gz` to `data/meld/video/`
- Extract `dev.tar.gz` to `data/meld/video/`
- Extract `test.tar.gz` to `data/meld/video/`

All `.mp4` video files should be in `data/meld/video/` folder.

**Step 3:** Copy CSV files:

- Rename `dev_sent_emo.csv` to `train.csv`
- Rename `test_sent_emo.csv` to `test.csv`
- Place all CSV files in `data/meld/text/` folder

**Final structure should look like:**

```
data/
└── meld/
    ├── text/
    │   ├── train.csv
    │   └── test.csv
    ├── video/
    │   ├── dia0_utt0.mp4
    │   ├── dia0_utt1.mp4
    │   └── ... (all video files)
    └── audio/
        └── (will be created by extract_audio.py)
```
## Model Architecture

**Text Branch:**

- BERT embeddings for contextual understanding
- BiLSTM for sequential dependencies
- Attention pooling for important words

**Audio Branch:**

- MFCC features (40 coefficients)
- CNN layers for temporal patterns
- BiLSTM for temporal dependencies
- Attention pooling for emotional cues

**Fusion:**

- Concatenate text and audio features
- Dense layers for joint representation
- Softmax classifier for emotion prediction

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
processing:
  sample_rate: 16000
  max_audio_length: 10
  n_mfcc: 40
  max_text_length: 128

model:
  text_bilstm_hidden: 256
  audio_bilstm_hidden: 256
  fusion_hidden: 512
  dropout: 0.3

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
```

**Dataset:**

- Training: ~9,000 utterances
- Test: ~2,600 utterances

## Project Timeline

| Month     | Tasks                                      |
| --------- | ------------------------------------------ |
| Sept 2025 | Dataset collection & analysis              |
| Oct 2025  | Data preprocessing pipeline                |
| Nov 2025  | Baseline model & comparative study         |
| Dec 2025  | Multimodal architecture design             |
| Jan 2026  | Model training & hyperparameter tuning     |
| Feb 2026  | Robustness testing & real-time integration |
| Mar 2026  | Final report preparation                   |
| Apr 2026  | Submit final report                        |

## References

1. Poria, S., et al. (2019). "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations." ACL.

2. Kim, S., & Lee, S. P. (2023). "A BiLSTM–Transformer and 2D CNN Architecture for Emotion Recognition from Speech." Electronics, 12(19), 4034.

3. Majumder, N., et al. (2019). "DialogueRNN: An Attentive RNN for Emotion Detection in Conversations." AAAI.

## Contact

**Student:** Anvitha Reddy Thupally  
**Student ID:** 304285735  
**Program:** Master's Project CSC 290

## License

This project is for academic purposes as part of a Master's thesis.
