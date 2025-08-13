# Heart Disease Analysis & Quiz App
## Overview
This project is an interactive heart disease analysis and quiz application built using Streamlit.
It uses the UCI Heart Disease dataset to perform Exploratory Data Analysis (EDA), visualize medical insights,
and engage users with a Kahoot-style cardiology quiz.

## Problem Statement
Heart disease remains one of the leading causes of death worldwide.
By analyzing patient medical records, we aim to:

Understand risk patterns

Identify key medical indicators

Encourage public awareness through an interactive quiz

## Research Questions
How is heart disease distributed among patients?

What is the relationship between maximum heart rate and age?

How is fasting blood sugar distributed?

How are major vessels distributed?

What is the distribution of resting blood pressure?

Are there gender differences in heart disease prevalence?

## Project Structure
bash
Copy
Edit
.
├── heart_disease_uci.csv     # Dataset
├── app.py                    # Streamlit main app
├── requirements.txt          # Dependencies
├── leaderboard.csv           # Auto-generated leaderboard file
├── README.md                 # Project documentation
├── screenshots/              # Folder containing app screenshots

## Data Dictionary
Column	Description
id	Unique patient ID
age	Age of patient (years)
origin	Place of study
sex	Male/Female
cp	Chest pain type
trestbps	Resting blood pressure (mm Hg)
chol	Serum cholesterol (mg/dl)
fbs	Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
restecg	Resting ECG results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina (1 = Yes, 0 = No)
oldpeak	ST depression induced by exercise
slope	Slope of the peak exercise ST segment
ca	Number of major vessels (0–3)
thal	Thalassemia type
num	Diagnosis of heart disease (0 = No, 1–4 = Yes)

## Features
EDA Dashboard:

Distribution plots for key variables

Relationship analysis between heart rate and age

Gender-based disease distribution

## Quiz Game:

5 questions for each difficulty level: Easy, Medium, Hard

Time-based scoring (faster answers = more points)

Saves leaderboard in CSV & PDF formats

Interactive Visualizations:

Seaborn & Matplotlib for static plots

Altair for interactive scatter plots

## Installation
bash
Copy
Edit
# Clone repository
git clone https://github.com/isaiahokumu/Heart-disease-analysis.git
cd Heart_disease

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
### Screenshots

### App Homepage

### EDA Dashboard

### Quiz Game

### Leaderboard PDF Example

(Screenshots should be placed in the screenshots/ folder in your project.)

## Example Visualizations
Heart disease distribution by gender

Age vs. maximum heart rate

Fasting blood sugar countplot

Resting blood pressure histogram

Major vessels distribution

## How the Quiz Works
Select your name and start the quiz.

Each question must be answered as quickly as possible for higher points.

Scores are calculated:

100 points max per question if answered instantly

Minimum 10 points if slow but correct

Final scores are stored in leaderboard.csv and a PDF leaderboard.
