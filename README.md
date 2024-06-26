# Binary Classification Project

This project aims to compare the performance of various supervised learning methods on a binary classification problem, which will help to understand the advantages and disadvantages of each classification algorithm.

## Project Overview

- The goal of this project is to build and evaluate different supervised learning models for binary classification.
- The dataset used in this project is `dataset_assignment1.csv`.
- The project compares the performance of K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM) classifiers.
- The project includes data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and model evaluation.

## Dataset

- The dataset used in this project is stored in the file `dataset_assignment1.csv`.
- It contains features and corresponding binary class labels.
- The dataset is split into training and testing sets with an 80/20 ratio.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

- `comp534_likhitha_mittoor.ipynb`: The main Python script that contains the code for data preprocessing, model training, and evaluation.
- `dataset_assignment1.csv`: The dataset file used in the project.
- `README.md`: This README file providing an overview of the project.

## Usage

1. Ensure that you have the required dependencies installed.
2. Place the `dataset_assignment1.csv` file in the same directory as the Python script.
3. Run the `comp534_likhitha_mittoor.py` script to execute the project.

## Results

- The project evaluates the performance of KNN, Random Forest, and SVM classifiers using various metrics such as accuracy, precision, recall, and F1-score.
- Hyperparameter tuning is performed using GridSearchCV to find the best hyperparameters for each model.
- The top scores for each model are displayed, along with the best hyperparameters.
- Learning curves are plotted to visualize the model's performance with increasing training examples.
- Confusion matrices are generated to assess the models' performance in terms of true positives, true negatives, false positives, and false negatives.
- The best model is selected based on the F1-score.
- Classification reports are generated for each model, providing detailed metrics for each class.
- A summary of the classification metrics is presented in a styled DataFrame.

## Conclusion

The project demonstrates the application of different supervised learning algorithms for binary classification. By comparing the performance of KNN, Random Forest, and SVM classifiers, we can gain insights into their strengths and weaknesses for this specific problem. The project showcases the importance of data preprocessing, hyperparameter tuning, and model evaluation in building effective classification models.

## Future Enhancements

- Experiment with additional classification algorithms and compare their performance.
- Explore feature selection techniques to identify the most informative features for classification.
- Investigate the impact of class imbalance on model performance and apply techniques to handle imbalanced datasets.
- Deploy the best-performing model as a web application or API for real-time predictions.