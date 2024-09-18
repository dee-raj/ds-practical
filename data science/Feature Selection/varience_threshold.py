"""
Variance is a measure of spread from the mean
If the varience is 0, then we have a feature with constant value
if a feature have a varience of 0, then it is likely not predictive
"""
import pandas  as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df['target'] = wine_data.target

# split data into train & test
X = wine_df.drop(['target'], axis=1)
Y = wine_df['target']

x_train, x_test, y_tarin, y_test = train_test_split(X, Y, shuffle=True, random_state=42, stratify=Y, test_size=0.3)



# Model : Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

# Initialize classifier
gbc = GradientBoostingClassifier(max_depth=5, random_state=42)

# Train Classifier using all features
gbc.fit(x_train, y_tarin)

# Make predictions
preds = gbc.predict(x_test)

# Evaluate the model using the f1-score
f1_score_all = round(f1_score(y_test, preds, average='weighted'), 4)
# print(f"F1 Score: {f1_score_all}")

accu_score = round(accuracy_score(y_test, preds), 4)
print(f"Accuracy Score: {accu_score}")


# Calculate the varience of each feature
variance = x_train.var(axis=0)
# print(variance)
"In this case none of the features are at same scale so the variance cam not be comapred"
"Because none of the variance in above is not equal to 0 or very very equal to 0"

from sklearn.preprocessing import MinMaxScaler
"By this we make variance 0 to 1 force-fully to analyse variance b/w them"
scaler = MinMaxScaler()
scaled_x_tarin = scaler.fit_transform(x_train)

scaled_variance = scaled_x_tarin.var(axis=0)
# print(scaled_variance)


"Bar plot for this scaled variance"
# '''
fig, ax = plt.subplots()
x = X.columns
y = scaled_variance

ax.bar(x,y, width=0.2)
ax.set_xlabel("Features")
ax.set_ylabel("Variances")
ax.set_ylim(0, 0.1)

for index, value in enumerate(y):
   plt.text(x=index, y=value+0.001, s=str(round(value, 3)), ha='center')

fig.autofmt_xdate()
plt.tight_layout()
# '''

selected_x_train = x_train.drop(['ash', 'magnesium'], axis=1)
selected_x_test = x_test.drop(['ash', 'magnesium'], axis=1)

gbc.fit(selected_x_train, y_tarin)
y_preds = gbc.predict(selected_x_test)

f1_score_selected = round(f1_score(y_test, y_preds, average='weighted'), 4)
# print(f1_score_selected)

"Bar plot for these selected features"
# '''
fig, ax = plt.subplots()

x=['All features', 'Varience threshold']
y=[f1_score_all, f1_score_selected]

ax.bar(x, y, width=0.3)
ax.set_xlabel("Features selection method")
ax.set_ylabel("F1-Score (weighted)")
ax.set_ylim(0, 0.5)

for index, value in enumerate(y):
   plt.text(x=index, y=value+0.001, s=str(round(value, 3)), ha='center')
plt.tight_layout(pad=1)
# '''