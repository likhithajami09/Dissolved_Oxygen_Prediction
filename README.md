# Smart and Sustainable Aquaculture: Predicting Dissolved Oxygen Using a Hybrid Deep Learning Approach
A complete intelligent system for predicting dissolved oxygen levels in aquaculture environments using a hybrid deep learning approach combined with machine learning techniques.

## Overview
This project presents a smart and sustainable aquaculture framework for accurate dissolved oxygen prediction. It integrates advanced machine learning and deep learning models to improve prediction accuracy and support real-time monitoring.

The system combines LightGBM-based feature selection with deep learning architectures such as RNN, GRU, BiLSTM, and Attention mechanisms to capture both short-term and long-term temporal dependencies in water quality data.

## Abstract
This system presents a Smart and Sustainable Aquaculture framework for dissolved oxygen prediction using a hybrid deep learning approach. The model combines LightGBM-based feature selection with RNN, GRU, BiLSTM, and an Attention mechanism to improve prediction accuracy in aquaculture environments. LightGBM identifies important water quality parameters and removes irrelevant features, reducing complexity and improving efficiency. The ensemble architecture captures temporal dependencies for reliable dissolved oxygen forecasting. A Flask web application with SQLite support is integrated for secure login, data upload, and testing. Experimental results show high accuracy and strong reliability, making the system effective for intelligent aquaculture management and sustainable water quality monitoring.

## Objective
- Identify important water quality parameters using LightGBM feature selection
- Improve accuracy of dissolved oxygen prediction
- Support real-time monitoring in aquaculture systems
- Enable intelligent decision-making for sustainable aquaculture

## Introduction
Aquaculture is growing rapidly, increasing the need for efficient water quality management. Dissolved oxygen is a critical factor for aquatic life. Low oxygen levels can lead to stress, disease, and reduced productivity.

Traditional monitoring methods are inefficient and fail to capture dynamic environmental changes. This system provides accurate prediction to help take early actions and improve sustainability.


## Existing System
- Uses models like Linear Regression, SVM, Decision Tree, Random Forest, ANN
- Hybrid models include LightGBM-LSTM and LightGBM-GRU
- Limited ability to capture long-term dependencies
- Lower accuracy and higher complexity

## Proposed System
- Uses an advanced hybrid ensemble model
- Combines:
  - LightGBM (feature selection)
  - RNN
  - GRU
  - BiLSTM
  - Attention mechanism
- Captures both short-term and long-term dependencies
- Provides better accuracy and efficiency

## Advantages
- Higher prediction accuracy
- Reduced computational complexity
- Effective feature selection
- Real-time monitoring capability
- Scalable and reliable system

## System Modules
1. Data Acquisition and Preprocessing
   - Collects water quality data (pH, temperature, turbidity, etc.)
   - Cleans and normalizes data

2. Feature Selection
   - Uses LightGBM to select important features

3. Prediction and Evaluation
   - Uses hybrid deep learning model
   - Evaluates using MSE, RMSE, MAE

4. User Interface
   - Flask-based web application
   - Secure login and data input

## Algorithm
1. LightGBM Feature Selection
   - Identifies important parameters
   - Removes irrelevant data

2. Hybrid Deep Learning Model
   - Combines BiLSTM, GRU, RNN, Attention
   - Captures temporal dependencies

3. Prediction
   - Uses Adam optimizer
   - Minimizes error using MSE

## Results
The system provides accurate dissolved oxygen prediction with strong performance and reliability.

## Comparative Study
| Model                          | RMSE     | MAE      | Accuracy |
|--------------------------------|----------|----------|----------|
| LightGBM-LSTM                  | 0.153118 | 0.121389 | 93.42%   |
| LightGBM-GRU                   | 0.144543 | 0.110472 | 94.11%   |
| LightGBM-BiSRU-Attention       | 0.125432 | 0.101289 | 96.28%   |
| Proposed Ensemble Model        | 0.109481 | 0.090655 | 97.85%   |

## Tech Stack
- Python
- Flask
- LightGBM
- Deep Learning Models (RNN, GRU, BiLSTM, Attention)
- Pandas
- Scikit-learn
- SQLite
- HTML, CSS, JavaScript

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run the application:
python app.py

3. Open browser:
http://127.0.0.1:5000/

## Project Structure
- Dataset/
- model/
- templates/
- static/
- uploads/
- app.py
- requirements.txt

## Conclusion
This system demonstrates the effectiveness of a hybrid ensemble model for dissolved oxygen prediction. It improves accuracy, reduces errors, and supports sustainable aquaculture management through intelligent monitoring and prediction.
