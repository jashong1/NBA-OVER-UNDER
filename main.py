# WNBA Over/Under Prediction Model
# Edited from NBA to WNBA since WNBA in currently in season and NBA is not. Will be able to test if model can accurate obtain profits long/short-term
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Find a way to download Datasets!!!!!
# Use basketball-reference for datasets
data = pd.read_csv('wnba_totals_sample.csv')

# Preview data
data.head()

# Select features and target
features = ['pace', 'off_eff', 'def_eff', 'home', 'rest_days', 'injuries']
X = data[features]
y = data['actual_total']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Visualize results
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel("Actual Totals")
plt.ylabel("Predicted Totals")
plt.title("WNBA Game Total Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Compare predictions to sportsbook line
X_test_copy = X_test.copy()
X_test_copy['Actual'] = y_test
X_test_copy['Predicted'] = predictions
X_test_copy['Sportsbook'] = data.loc[X_test.index, 'sportsbook_total']

# Define bet decision function
def bet_decision(row):
    diff = row['Predicted'] - row['Sportsbook']
    if diff > 8:
        return 'Bet Over'
    elif diff < -8:
        return 'Bet Under'
    else:
        return 'No Bet'

X_test_copy['Decision'] = X_test_copy.apply(bet_decision, axis=1)

# Show results
X_test_copy[['Actual', 'Predicted', 'Sportsbook', 'Decision']].head(10)
