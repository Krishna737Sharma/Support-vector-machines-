# Support Vector Machines

## Objective
To compare the performance of Support Vector Regressors using Polynomial and RBF kernels on the UCI Real Estate Valuation dataset.

## Steps Performed
1. Loaded the dataset and explored its structure.
2. Preprocessed the data:
   - Removed duplicates
   - Handled missing values
   - Scaled numerical features
3. Split the dataset into training, validation, and test sets.
4. Trained SVRs using Polynomial and RBF kernels.
5. Used GridSearchCV for hyperparameter tuning:
   - Polynomial degree
   - RBF kernel gamma
6. Plotted:
   - MSE vs. Polynomial degree
   - MSE vs. Gamma for RBF kernel
7. Compared performance and visualized predictions vs. ground truth.

## Results
- Best Polynomial Degree: `<value>`
- Best Gamma for RBF Kernel: `<value>`
- Scatter plot shows the prediction accuracy.

## How to Run
1. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib
