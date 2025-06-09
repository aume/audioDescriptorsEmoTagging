# Soundscape Emotion Prediction and Tagging System

This project implements a comprehensive pipeline for analyzing audio, predicting their Valence and Arousal and HLD escriptor dimensions using machine learning, and then tagging them with specific HLD descriptors based on these predictions. Finally, all results, including raw features, predicted scores, and descriptor tags, are logged into a relational SQLite database for easy querying and analysis.

https://www.metacreation.net/projects/emo-soundscapes/

A Survey of High-Level Descriptors in Sound Design: Understanding the Key Audio Aesthetics Towards Developing Generative Game Audio Engines. C Anderson, C Carpenter, J Kranabetter, M Thorogood - Audio Engineering Society Conference: AES 2024


https://saifmohammad.com/WebPages/nrc-vad.html

## Overview

The system processes audio files in a multi-stage pipeline:
1.  **Feature Extraction**: Low-level audio features are extracted from raw WAV files using the Essentia library.
2.  **Model Training**: Separate Support Vector Regression (SVR) models are trained for Valence and Arousal prediction. These models include feature scaling and selection steps.
3.  **Emotion Prediction**: The trained models are used to predict Valence and Arousal scores for new, unseen audio files.
4.  **Descriptor Tagging**: Based on the predicted Valence and Arousal, a custom algorithm calculates localized scores for predefined emotional descriptor pairs (e.g., "soothing/alarming", "happy/sad").
5.  **Database Logging**: All extracted features, predicted VA scores, and descriptor scores are systematically stored in a SQLite database.

## System Components

The project is modularized into several Python scripts for clarity and maintainability:

* **`train_emo_model.py`**:
    * **Role**: The primary script for **training** the Valence and Arousal prediction models.
    * **Relationship**: It orchestrates data loading, feature extraction (via `extractor.py`), model training (using `sklearn.pipeline`), model saving, and initial evaluation. It depends on `va_data_loader.py`, `extractor.py`, `va_plotter.py`.
    * **Output**: Generates `out_Valence.csv` and `out_Arousal.csv` (containing extracted features), and saves `trained_valence_model.joblib` and `trained_arousal_model.joblib` (the fitted pipelines, including learned feature lists).

* **`va_visualize_descriptors.py`**:
    * **Role**: Visualize the emo dataset and descriptor pairs in VA space
    * **Relationship**: from train_emo_model.py - VAD-Lexicon, descriptorPairs, out_Valence, out_Arousal, va_point_descriptor_scores
    * **Output**: Figure plotting emo dataset and descriptor pairs in VA space

* **`predict_new_audio.py`**:
    * **Role**: Handles **prediction** of Valence/Arousal and descriptor tagging for new, unseen audio files.
    * **Relationship**: Loads the trained models from `train_emo_model.py`'s output, extracts features from new audio (via `extractor.py`), performs predictions, and calculates descriptor scores (via `va_analyzer.py`).
    * **Output**: Generates `new_audio_features_raw.csv` (raw features for new audio), `new_audio_predicted_va.csv` (predicted VA scores), and `new_audio_descriptor_scores.csv` (descriptor scores).

* **`log_predictions_to_db.py`**:
    * **Role**: Logs all generated data (raw features, predicted VA, descriptor scores) into a relational SQLite database.
    * **Relationship**: Reads CSV outputs from `predict_new_audio.py` and populates a SQLite database with structured tables.
    * **Output**: Creates or updates `audio_predictions.db`.

* **`extractor.py`**:
    * **Role**: Implements the logic for extracting low-level acoustic features from individual audio files.
    * **Relationship**: Used by both `train_emo_model.py` (for training data) and `predict_new_audio.py` (for new data). It depends on `essentia_engine.py` for core Essentia functionality.

* **`essentia_engine.py`**:
    * **Role**: A wrapper for Essentia's `Extractor` algorithm, configuring its parameters for low-level feature extraction.
    * **Relationship**: Directly utilized by `extractor.py`.

* **`va_utils.py`**:
    * **Role**: Contains mathematical utility functions, such as Euclidean distance calculation and the core logic for localized descriptor scoring.
    * **Relationship**: Imported and used by `va_analyzer.py`.

* **`va_data_loader.py`**:
    * **Role**: Provides robust functions for loading various data files: VAD lexicons, descriptor pairs, and input/output VA prediction CSVs.
    * **Relationship**: Used by `train_emo_model.py` and `predict_new_audio.py`.

* **`va_analyzer.py`**:
    * **Role**: Implements the algorithm for calculating localized descriptor scores based on predicted Valence-Arousal coordinates.
    * **Relationship**: Used by `train_emo_model.py` (for analysis) and `predict_new_audio.py` (for tagging).

* **`va_plotter.py`**:
    * **Role**: Contains functionality for visualizing VA data and descriptor pairs.
    * **Relationship**: Used by `train_emo_model.py` to plot training data.

* **`train_descriptor_model.py`**:
    * **Role**: Trains, visualizes, and saves model per descriptor
    * **Relationship**: Uses audio_predictions.db

* **`train_and_visualize_all_descriptors.py`**:
    * **Role**: Trains and visualizes all descriptor models 
    * **Relationship**: Uses audio_predictions.db

## Prerequisites

Before running the system, ensure you have the following installed:

* **Python 3.8+** (recommended to use a virtual environment)
* **pip** (Python package installer)
* **Essential Python Libraries**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib joblib librosa soundfile essentia
    ```
    * `librosa` and `soundfile` are needed for audio processing in `extractor.py`.
    * `essentia` is the core audio feature extraction library. Ensure it's correctly installed for your OS/architecture (sometimes requires specific instructions or a pre-compiled wheel).
* **FFmpeg**: Essentia and Librosa often rely on FFmpeg for reading and decoding various audio formats.
    * **macOS**: `brew install ffmpeg`
    * **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
    * **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

## Dataset Structure

The system expects a specific directory structure for your data. You should place your dataset files as follows (relative to where you run `main.py` and `predict_new_audio.py`):