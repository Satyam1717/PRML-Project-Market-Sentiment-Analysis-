# Market Sentiment Analysis for Stock Price Prediction

![GitHub](https://img.shields.io/badge/Python-3.7%2B-blue)
![GitHub](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

This project analyzes market sentiment from tweets and financial headlines related to Reliance Industries to forecast stock prices using time series models. By combining natural language processing techniques with time series forecasting, we developed a model that can predict future stock prices based on market sentiment.

## 🚀 Live Demo

Check out our live Streamlit app: [Market Sentiment Analysis App](https://streamlit-app-228333235452.asia-south1.run.app/)

## 🛠 Tech Stack

- **Python Libraries**: NumPy, Pandas, Scikit-learn, NLTK
- **Time Series Forecasting**: SARIMAX from statsmodels
- **Word Embeddings**: Word2Vec
- **Machine Learning**: XGBoost
- **Data Visualization**: Matplotlib, Plotly
- **Deployment**: Streamlit

## 📊 Project Structure

```
├── data_scrapping.ipynb             # Data collection from online sources
├── filtering_articles.ipynb         # Filtering relevant articles
├── preprocessing_detailed_analysis.ipynb  # Data preprocessing steps
├── sentiment_score_prediction.ipynb # Sentiment analysis implementation
├── sarima_test_train.ipynb          # SARIMAX model training and testing
├── w2v+xgb.ipynb                    # Word2Vec and XGBoost implementation
├── final_prediction.ipynb           # Final model evaluation
├── stock.ipynb                      # Stock data collection and analysis
├── main.py                          # Streamlit app implementation
├── requirements.txt                 # Dependencies
├── dt/                              # Directory containing articles datasets
│   ├── banking_finance_articles.csv
│   ├── corporate_trends.csv
│   └── [other sector articles]...
└── trained_models/                  # Saved trained models
    ├── ewma_model.pkl
    ├── stock_model.pkl
    ├── word2vec_model.model
    └── xgboost_model.pkl
```

## 📋 Methodology

Our approach integrates natural language processing with time series forecasting to predict stock prices based on market sentiment. The methodology consists of the following comprehensive steps:

### 1. Data Collection and Preparation

- **Multi-source Data Scraping**: Systematically collected financial news articles and tweets related to Reliance Industries from various sources using web scraping techniques
- **Sector-specific Categorization**: Organized articles into distinct sectors (banking/finance, energy/oil/gas, telecom, retail, etc.) as seen in the `dt/` directory
- **Stock Data Acquisition**: Retrieved historical stock price data for Reliance Industries with OHLC (Open, High, Low, Close) prices and trading volumes
- **Data Alignment**: Synchronized news articles and tweets with corresponding stock price data based on publication dates

### 2. Data Preprocessing and Text Analysis

- **Text Normalization**:
  - Removed HTML tags, special characters, and URLs
  - Converted text to lowercase
  - Eliminated redundant whitespace
- **Advanced Text Cleaning**:
  - Applied lemmatization to reduce words to their base forms
  - Removed stop words using NLTK's stopwords corpus
  - Filtered out non-informative tokens and rare terms
- **Domain-specific Filtering**: Retained finance-specific terminology that might be filtered by generic preprocessing
- **Feature Engineering**: Created temporal features to capture market cycles and seasonality patterns

### 3. Sentiment Analysis Implementation

- **Word Embeddings**:
  - Built custom Word2Vec models trained on financial news corpus
  - Used 500-dimensional vectors to capture semantic relationships between financial terms
- **TF-IDF Weighting**:
  - Calculated Term Frequency-Inverse Document Frequency to weigh the importance of words
  - Generated TF-IDF weighted document vectors by combining Word2Vec embeddings with TF-IDF scores

### 4. Temporal Processing and Signal Smoothing

- **Time Series Aggregation**: Consolidated multiple daily news sources into single daily sentiment scores
- **Signal Processing**:
  - Implemented Exponentially Weighted Moving Average (EWMA) with multiple span values (3, 5, 10, 15) to smooth sentiment volatility
  - Compared different span values' effectiveness in capturing sentiment trends while reducing noise
  - Selected optimal EWMA span (3) based on correlation with stock price movements
- **Feature Comparison**: Visualized raw sentiment scores against smoothed EWMA values to identify optimal signal processing parameters

### 5. Model Development and Training

- **SARIMAX Model Development**:
  - Incorporated exogenous sentiment variables (EWMA scores) into the SARIMAX framework
  - Applied grid search to optimize SARIMA parameters (p,d,q) × (P,D,Q)s for seasonal components
  - Conducted stationarity tests to ensure proper time series properties
- **Alternative ML Approach**:
  - Implemented XGBoost regression with Word2Vec embeddings as an alternative prediction technique

### 6. Model Evaluation and Validation

- **Performance Metrics**:
  - Calculated Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE)
  - Measured statistical significance using p-values for exogenous variables
  - Conducted residual analysis to verify model assumptions
- **Comparative Analysis**:
  - Benchmarked against traditional time series models without sentiment variables
  - Performed out-of-sample testing on unseen data periods
  - Evaluated directional accuracy (predicting price movements up/down correctly)
- **Visualization**:
  - Generated prediction intervals with 95% confidence bounds
  - Created interactive visualizations of predicted vs. actual prices

### 7. Deployment Pipeline

- **Model Persistence**:
  - Saved trained models (SARIMAX, XGBoost) using pickle serialization
  - Stored Word2Vec embeddings for production use
- **Interactive Web Application**:
  - Developed a Streamlit application with real-time prediction capabilities
  - Implemented visualization components for trend analysis and prediction confidence intervals
  - Created user-friendly interface for viewing predictions and sentiment trends

## 📈 Results

Our model showed meaningful correlation between sentiment and stock movement. The forecasts closely matched real data during evaluation, with the following metrics:

- Mean Absolute Percentage Error (MAPE): ~2.9738%
- Root Mean Squared Error (RMSE): ~43.9098

## Project Workflow

[Market Sentiment Analysis Workflow](flowchart.png)


## 🚀 Getting Started

### Prerequisites

- Python 3.7 or above
- pip package manager

### Installation

1. Clone this repository:

```bash
git clone https://github.com/sagar7162/PRML-Project-Market-Sentiment-Analysis-.git
cd PRML-Project-Market-Sentiment-Analysis-
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run main.py
```

## 🔍 Future Improvements

- Incorporate additional data sources for sentiment analysis
- Experiment with transformer-based models for improved text understanding
- Implement real-time tweet analysis for more timely predictions
- Add support for analyzing multiple stocks

## 👥 Contributors

- Satyam Jha, Krish Jain, Sagar Ratna Chaudhary, Nishchal Badaya, Shivam, Avni Gupta
- PRML Project | B.Tech 2023 Batch

## 🔗 Links

- [GitHub Repository](https://github.com/sagar7162/PRML-Project-Market-Sentiment-Analysis-#)
- [Live Demo](https://streamlit-app-228333235452.asia-south1.run.app/)
