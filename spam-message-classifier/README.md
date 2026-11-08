# SMS Spam Message Classifier

An interactive SMS spam detection application built with Python, Streamlit, scikit-learn, NLTK, Pandas and Matplotlib. It classifies messages as **Spam** or **Ham (Not Spam)** using TF-IDF features and a Multinomial Naive Bayes model.

## Features
- Classify a single SMS and view spam/ham probabilities.
- Classify multiple messages and download results as CSV.
- View accuracy, precision, recall, F1-score and a confusion matrix.
- Explore class distribution and words associated with each class.
- Upload a labelled CSV to train the model.
- Attempts to load the UCI SMS Spam Collection; uses a bundled sample if offline.

## Project structure
```text
spam-message-classifier/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_messages.csv
└── src/
    ├── __init__.py
    └── classifier.py
```

## Run on Windows (VS Code)
Open this folder in VS Code and run:

```powershell
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501` if the browser does not open automatically.

If the environment and packages are already installed, run only:
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## Custom dataset
Upload a CSV with columns named `label` and `message`. Labels must be `ham` or `spam`. At least 20 valid rows and both classes are required.

## Dataset
The app attempts to download the UCI SMS Spam Collection when online. If unavailable, it uses `data/sample_messages.csv`, a small bundled demonstration dataset. Metrics from this fallback sample are not representative benchmark results.

Original dataset: Almeida, T. A. and Hidalgo, J. M. (2011), SMS Spam Collection, UCI Machine Learning Repository, https://doi.org/10.24432/C5CC84.

## Note
This is an educational baseline, not a security product. Predictions can be wrong; do not use them as the sole basis for security decisions.
