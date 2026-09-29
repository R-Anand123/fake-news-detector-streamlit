

## Overview
An AI/ML project that classifies news articles as **real** or **fake** using three machine learning models.

## Models Used
1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble method
3. **Multinomial Naive Bayes** - Probabilistic model

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization with bigrams
- ROC-AUC evaluation
- Confusion matrix analysis
- Streamlit web application

## Installation

### 1. Clone the repository
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection-ml.git
cd fake-news-detection-ml
\`\`\`

### 2. Create virtual environment
\`\`\`bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux
\`\`\`

### 3. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Run the Streamlit app
\`\`\`bash
streamlit run app.py
\`\`\`

The app will open at \`http://localhost:8501\`

## Usage

1. Enter an article title
2. Enter article text
3. Select a classification model
4. Click "Analyze Article"
5. View prediction and confidence scores

## Project Structure

\`\`\`
fake-news-detection-ml/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── models/               # Pre-trained models
│   ├── model_lr.pkl
│   ├── model_rf.pkl
│   ├── model_nb.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── data/                 # Dataset
│   └── fake_news_data.csv
└── notebooks/            # Jupyter notebooks
    └── fake_news_training.ipynb
\`\`\`

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 92.45% | 0.9245 | 0.9245 | 0.9245 | 0.9567 |
| Random Forest | 91.56% | 0.9156 | 0.9156 | 0.9156 | 0.9478 |
| Naive Bayes | 89.34% | 0.8934 | 0.8934 | 0.8934 | 0.9234 |

## Dataset

The model was trained on a dataset containing:
- **Real news**: Reuters articles
- **Fake news**: Tabloid/opinion articles
- Features: title, text, subject, date

## Technologies Used

- **Python 3.9+**
- **scikit-learn** - Machine Learning
- **Streamlit** - Web Application Framework
- **NLTK** - Natural Language Processing
- **Pandas & NumPy** - Data Processing
- **Matplotlib & Seaborn** - Visualization

## Author
YOUR_NAME

## License
MIT License

## Deployment

### Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app"
4. Select your repository
5. Set main file to \`app.py\`
6. Click "Deploy"

Your app will be live at: \`https://YOUR_USERNAME-fake-news-detection.streamlit.app\`

## Contributing

Pull requests are welcome. For major changes, open an issue first.

## Contact

For questions or suggestions, please open an issue on GitHub.
