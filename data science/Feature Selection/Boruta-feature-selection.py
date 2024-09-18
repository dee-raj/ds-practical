"""
 Boruta 
 - Take the human out
 - A feature selection algorithm where we do not need to set a subjective thresold
 
  1. Create shadow features: randomly shuffled version of each feature
  2. Compute the importance of each shadow feature -> the highest score becomes our threshold
  3. If the importance of the original feature is higher than the threshold -> keep (otherwise -> discard)
"""
"""
Boruta Datasets:- 
   Feature 1,      Feature 2,        feature 3
      1               4                 7
      2               5                 8
      3               6                 9

Boruta Shadow:-
   Feature 1,      Feature 2,        feature 3,     shadow_1,           shadow_2,             shadow_3
      1               4                 7              3                  6                     9
      2               5                 8              1                  4                     8
      3               6                 9              2                  5                     7
   
Boruta Importance:- 
   Feature 1,      Feature 2,        feature 3,     shadow_1,           shadow_2,             shadow_3
     I=4               I=12            I=11            I=2                 I=10                 I=5
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
wine_df["target"] = wine_data.target

X = wine_df.drop(["target"], axis=1)
Y = wine_df["target"]
x_tarin, x_test, y_train, y_test = train_test_split(
   X, Y, random_state=42, test_size=0.3, stratify=Y, shuffle=True
)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbc = GradientBoostingClassifier(max_depth=5, random_state=42)
gbc.fit(x_tarin, y_train)

y_preds = gbc.predict(x_test)
accuracy = round(accuracy_score(y_test, y_preds), 4)
print("Original Accuracy: ",accuracy)


from boruta import BorutaPy

boruta_selector = BorutaPy(gbc, random_state=1)
boruta_selector.fit(x_tarin.values, y_train.values.ravel())

selected_x_train = boruta_selector.transform(x_tarin.values)
selected_x_test = boruta_selector.transform(x_test.values)

gbc.fit(selected_x_train, y_train)

boruta_preds = gbc.predict(selected_x_test)
boruta_accuracy = round(accuracy_score(y_test, boruta_preds),4)


selected_features_mask = boruta_selector.support_

selected_features = x_tarin.columns[selected_features_mask]
print(selected_features)

"Ploting Bar graph"
fig, ax = plt.subplots()

x = ["original features", "Boruta features", "Filter", "RFE", "Variance threshold"]
y = [accuracy, boruta_accuracy, 0.981, 1.0, 0.948]

ax.bar(x, y, width=0.2)
ax.set_xlabel("Feature selection methods")
ax.set_ylabel("Accuracy of models")
ax.set_ylim(0, 1.2)

for idx, val in enumerate(y):
   plt.text(x=idx + 1, y=val + 0.001, s=str(val), ha="center")

plt.tight_layout()
