# Seq2Fold: Protein Fold Prediction using Machine Learning (SVM)

## Project Overview

Seq2Fold predicts the structural fold of a protein from its amino acid sequence using Machine Learning, specifically a Support Vector Machine (SVM).

Proteins are made of amino acid sequences, and their 3D structure, or fold, plays an important role in determining their biological function. This project uses sequence-based features to classify protein sequences into fold classes.

## Features

- Predict protein fold from amino acid sequence
- Supports manual sequence input
- Supports FASTA file upload
- Batch prediction for multiple sequences
- Displays confidence score
- Shows confidence distribution graph
- Allows CSV download for FASTA prediction results
- Simple Streamlit web interface

## Input

### Manual Input

Enter a protein sequence directly in the app:

```text
SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHANMGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV
```
FASTA File Input
Upload a .fa, .fasta, or .txt FASTA file:

>protein1
SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHANMGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV

>protein2
QAIPMTLRGAEKLREELDFLKSVRRPEIIAAIAEAREHGDLKENAEYHAAREQQGFCEGRIKDIEAKLSNAQVIDVTK
Methodology
Step 1: Feature Extraction
The protein sequence is converted into numerical features.

AAC: Amino Acid Composition
DPC: Dipeptide Composition
Total features: 420
AAC captures the frequency of each amino acid.

DPC captures the frequency of amino acid pairs.

Step 2: Feature Scaling
The extracted features are standardized using StandardScaler.

Step 3: Dimensionality Reduction
PCA is applied to reduce the feature size.

420 features -> 150 PCA components
This helps reduce noise and improves model performance.

Step 4: Model Training
The model is trained using:

Algorithm: Support Vector Machine
Kernel: RBF
Dataset: SCOP / ASTRAL protein dataset
Output
The app displays:

Predicted fold class
Confidence score
Confidence distribution graph
Batch prediction table for FASTA files
Downloadable CSV result file
Current Fold Classes
The current trained model predicts only these fold classes:

1, 2, 4, 37, 58
To predict more fold classes, the model must be retrained using more fold labels.

Project Structure
protein_svm_project/
│
├── data/
│   ├── astral_40.fa
│   ├── fold_data.csv
│   ├── full_data.csv
│   └── homology_data.csv
│
├── models/
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── selector.pkl
│
├── notebooks/
│   └── colab_training.ipynb
│
├── src/
│   ├── feature_extraction.py
│   ├── prediction_utils.py
│   ├── prepare_data.py
│   ├── train_svm.py
│   └── predict.py
│
├── app.py
├── test_different_folds.fasta
├── requirements.txt
└── README.md

How to Run
1. Install Dependencies
pip install -r requirements.txt
2. Train the Model
cd src
python train_svm.py
3. Run the App
cd ..
streamlit run app.py
Then open the local URL shown in the terminal.

Usually:

http://localhost:8501
Test FASTA File
A sample FASTA file is included:

test_different_folds.fasta
Expected predictions:

fold_1_example  -> 1
fold_2_example  -> 2
fold_4_example  -> 4
fold_37_example -> 37
fold_58_example -> 58
Results
Model performance may vary depending on the training data and sequence similarity.

Current approximate accuracy:

60% - 65%
Performance depends on:
Sequence complexity
Similarity between folds
Dataset balance
Feature representation
