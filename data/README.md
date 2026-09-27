# Advertising Sales Prediction using Machine Learning

## Project Overview

This project predicts product sales based on advertising expenditure using Machine Learning.

The project uses Linear Regression to predict Sales from three advertising channels:

- TV Advertising
- Radio Advertising
- Newspaper Advertising

A Streamlit web application is created to allow users to enter advertising expenditure and get predicted sales.

## Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| TV | Advertising expenditure on TV |
| Radio | Advertising expenditure on Radio |
| Newspaper | Advertising expenditure on Newspaper |
| Sales | Product sales |

The dataset used in this project contains 106 records.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked for missing values.
3. Removed duplicate rows.
4. Selected TV, Radio and Newspaper as input features.
5. Selected Sales as the target variable.
6. Split the dataset into training and testing data using an 80:20 ratio.

## Model Building

A Linear Regression model was used for prediction.

The model was trained using:

- TV
- Radio
- Newspaper

Target variable:

- Sales

## Model Evaluation

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Evaluation Results

| Metric | Value |
|---|---:|
| MAE | 1.6173 |
| MSE | 4.0618 |
| RMSE | 2.0154 |
| R² Score | 0.8332 |

The R² score of 0.8332 indicates that the model explains approximately 83.32% of the variation in the test-set sales values.

## Model Saving

The trained Linear Regression model was saved using Joblib as:

`model.pkl`

## Streamlit Application

The Streamlit application allows users to enter:

- TV Advertising Expenditure
- Radio Advertising Expenditure
- Newspaper Advertising Expenditure

The application then predicts the expected sales.

Example:

TV = 100  
Radio = 20  
Newspaper = 20  

Predicted Sales = 12.64

## Prediction and Interpretation

The model uses the entered advertising expenditure values to estimate product sales.

Higher or lower advertising expenditure across TV, Radio and Newspaper can result in different predicted sales values according to the patterns learned by the Linear Regression model.

## Project Structure

```text
Advertising-Sales-Prediction/
│
├── data/
│   └── advertising.csv
│
├── screenshots/
│   ├── streamlit_prediction.png
│   ├── model_evaluation.png
│   └── project_structure.png
│
├── app.py
├── train_model.py
├── model.pkl
└── README.md