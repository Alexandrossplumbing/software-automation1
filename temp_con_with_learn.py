import numpy as np
from sklearn.linear_model import LinearRegression

# Generate training data: random Celsius values
np.random.seed(42)
celsius_values = np.random.uniform(-20, 100, 100).reshape(-1, 1)
fahrenheit_values = celsius_values * 1.8 + 32  # True formula, used to label the data

# Train a linear regression model
model = LinearRegression()
model.fit(celsius_values, fahrenheit_values)

# Output the learned formula
print("\nML-based conversion model:")
print(f"Learned slope (should be ~1.8): {model.coef_[0][0]:.4f}")
print(f"Learned intercept (should be ~32): {model.intercept_[0]:.4f}")

# Example prediction
example = 25
predicted = model.predict([[example]])[0][0]
print(f"{example}°C ≈ {predicted:.2f}°F (predicted)")


