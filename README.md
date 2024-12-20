# -Price-Prediction
It is useful for pricing house prices

# House Price Prediction using Linear Regression

This project demonstrates a simple implementation of house price prediction using linear regression with Python. It uses sample data to train a linear regression model and predicts the price of a house based on its size in square feet.

## Overview

Linear regression is a basic yet powerful technique for modeling relationships between dependent and independent variables. In this project, we use the house size (in square feet) as the independent variable and the house price (in thousands of dollars) as the dependent variable.

## Access My Colab
```
https://colab.research.google.com/drive/1p6l2EZYkkBn_A2s3LOmmJeil9NBXkbug?usp=drive_link
```

## Code Explanation

### Libraries Used
- `numpy`: For numerical computations and data handling.
- `sklearn.linear_model`: Provides the `LinearRegression` class to perform linear regression.

### Steps
1. **Data Preparation**:
    - `X` represents the sizes of houses (in square feet).
    - `y` represents the corresponding prices of houses (in thousands of dollars).
2. **Model Creation**:
    - A linear regression model is instantiated using `LinearRegression()`.
3. **Model Training**:
    - The model is trained on the provided dataset using the `fit` method.
4. **Prediction**:
    - The trained model predicts the price of a house with a given size (e.g., 2000 square feet).
5. **Result Output**:
    - The predicted price is printed.

## Code
```python
# Import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data: House size (in square feet) and corresponding prices (in thousands of dollars)
X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]).reshape(-1, 1)
y = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

# Create a linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict the price of a house with size 2000 square feet
predicted_price = model.predict([[2000]])

# Print the predicted price
print("Predicted price for a 2000 sq. ft. house:", predicted_price[0])
```

## Example Output
```
Predicted price for a 2000 sq. ft. house: 306.93763440860213
```

## How to Use
1. Clone the repository.
2. Install the necessary dependencies (e.g., `numpy`, `scikit-learn`).
3. Run the script to see the predicted house price.

## Future Enhancements
- Use a larger, real-world dataset for better predictions.
- Add data preprocessing steps like normalization and handling missing values.
- Implement multiple regression to include additional features like the number of bedrooms, location, etc.

---
Feel free to contribute or raise issues for further improvements!

