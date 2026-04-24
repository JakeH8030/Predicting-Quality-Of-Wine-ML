# Predicting Wine Quality: A Stacking Ensemble Approach

This project is a comprehensive machine learning analysis that predicts the quality of red and white wines. It focuses on utilizing advanced techniques, such as stacking ensembles, to effectively overcome severe class imbalances in the dataset.

## Project Overview
The goal of this project is to classify wines into four distinct categories (`white_high`, `white_low`, `red_high`, `red_low`) based on their physicochemical properties. A significant challenge in this dataset is the severe class imbalance (e.g., a 4.38:1 ratio in certain classes), requiring specialized evaluation metrics and ensemble modeling strategies to ensure minority classes are predicted accurately.

## Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* **Original Study:** P. Cortez et al., *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems (2009).
* **Data:** Two datasets (`winequality-red.csv` and `winequality-white.csv`) representing Portuguese *Vinho Verde* wines.
* **Features:** 11 numerical physicochemical tests (e.g., volatile acidity, citric acid, alcohol content) and a sensory quality score (0-10).

## Methodology & Approach
1. **Data Preprocessing & Feature Engineering:**
   * Merged the red and white wine datasets.
   * Engineered a new 4-class target variable combining wine type (Red/White) and binned quality (High/Low).
   * Addressed and quantified the class imbalance.
2. **Exploratory Data Analysis (EDA):**
   * Visualized quality distributions using precisely binned, automated percentage-annotated histograms.
   * Identified systematic rating shifts between red and white wines.
3. **Model Evaluation Strategy:**
   * Transitioned away from standard accuracy due to class imbalance.
   * Prioritized **Balanced Accuracy**, **F1-Score**, and **Class-specific Recall** to evaluate model performance fairly.
4. **Machine Learning Modeling:**
   * **Base Models:** Trained and evaluated K-Nearest Neighbors (K-NN) and Support Vector Classifier (SVC).
   * **Stacking Ensemble (Meta-Learner):** Architected a stacking ensemble model designed to patch individual model weaknesses. By learning which model to trust for specific predictions, the meta-learner successfully leveraged the SVC's strength on minority classes to compensate for K-NN's blind spots.

## Key Results
* Successfully handled a complex 4-class classification problem with a high imbalance ratio.
* The baseline K-NN model struggled with the minority `white_low` class (Recall: 0.64).
* The final **Stacking Ensemble model** boosted the recall for the `white_low` class to an excellent **0.80**, creating a robust final model that outperformed individual base models.

## Tech Stack & Tools
* **Languages:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn matplotlib seaborn jupyter`
3. Download the `winequality-red.csv` and `winequality-white.csv` files from the UCI repository and place them in the root directory.
4. Open and run the Jupyter Notebook:
