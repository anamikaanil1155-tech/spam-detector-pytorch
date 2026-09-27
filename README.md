# NLP Text Classifier — Spam Detection with PyTorch

A deep learning NLP pipeline that classifies text messages as **Spam** or **Ham** (not spam) using TF-IDF vectorization and a multi-layer neural network built in PyTorch.

## Overview

This project implements an end-to-end text classification pipeline:

1. **Data generation** — 1,000 labeled messages built from randomized templates (no external download required, fully reproducible)
2. **Vectorization** — TF-IDF with unigrams + bigrams (3,000 features)
3. **Model** — a multi-layer feed-forward neural network with `Linear → BatchNorm → ReLU → Dropout` blocks
4. **Training** — full train/validation loop with **early stopping** on validation loss
5. **Evaluation** — accuracy, precision/recall/F1, and a confusion matrix on a held-out test set
6. **Inference** — predictions with confidence scores on brand-new, unseen sample inputs

## Project Structure

```
.
├── text_classifier_nlp_pytorch.ipynb   # Main notebook (data → training → evaluation → inference)
├── README.md
└── training_curves.png                 # Saved automatically when the notebook is run
```

## Requirements

- Python 3.9+
- torch
- scikit-learn
- pandas
- numpy
- matplotlib

Install everything with:

```bash
pip install torch scikit-learn pandas numpy matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies (see above).
3. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook text_classifier_nlp_pytorch.ipynb
   ```
   Or open it directly in [Google Colab](https://colab.research.google.com/) — no local setup needed.
4. Running all cells will:
   - Generate/load the dataset and vectorize it
   - Train the model with early stopping
   - Plot and save loss/accuracy convergence curves
   - Print test-set evaluation metrics
   - Run inference on sample unseen messages with confidence scores

## Model Architecture

```
Input (TF-IDF vector, 3000-dim)
   → Linear(3000, 256) → BatchNorm1d → ReLU → Dropout(0.3)
   → Linear(256, 64)   → BatchNorm1d → ReLU → Dropout(0.3)
   → Linear(64, 2)     → logits (Ham / Spam)
```

Trained with `CrossEntropyLoss` and the `Adam` optimizer (`lr=1e-3`, `weight_decay=1e-5`), with early stopping (patience = 6 epochs) restoring the best validation-loss checkpoint.

## Example Output

After training, the notebook reports test accuracy, a full classification report, and runs inference on unseen inputs, e.g.:

| Text | Prediction | Confidence |
|---|---|---|
| "Congratulations, you've been chosen to win a brand new car, click here now!!!" | Spam | 98.7% |
| "Hi mom, just checking in - I'll call you tonight after dinner." | Ham | 97.4% |

*(Exact numbers will vary slightly by run/seed.)*

## Using a Real Dataset

The synthetic dataset can be swapped for any real labeled text dataset (e.g. [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), IMDB reviews for sentiment analysis) by replacing the data-generation cell with:

```python
df = pd.read_csv("your_dataset.csv")  # expects 'text' and 'label' columns
```

Everything downstream — vectorizer, PyTorch `Dataset`/`DataLoader`, model, training loop, evaluation, and inference — works unchanged.

## Possible Extensions

- Replace TF-IDF with Hugging Face tokenizers + pretrained embeddings (e.g. `bert-base-uncased`) for stronger performance
- Add k-fold cross-validation
- Experiment with deeper architectures, different dropout rates, or learning-rate schedules
- Deploy the trained model behind a simple Flask/FastAPI endpoint

## License

MIT — feel free to use, modify, and share.
