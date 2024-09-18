"""
Recursive feature elimination (RFE)
- A wrapper method that uses the model to calculate the feature importance
- Must use a model that has a way of calculating feature importance (tree-based features)
- eg:- random forest, decision tree, ...
"""

# Algorithm
"""                                   Yes
                        |--<-------------<------------|
                        |                             |
                 |------------------|         |----------------|    No
All features --->| Remove "weakest" | ------> | # features > k |----------> Done
                 |  variable        |         |                |
                 |------------------|         |----------------|

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
def acc_p(k:int)->None:
   if k == 1:
      print(f"Found accuracy of {k} feature.")
   elif k < 13:
      print(f"Found accuracy of {k} features. please wait few more")
   else:
      print(f"Found accuracy of {k} features. Printing Here By")
   

from sklearn.feature_selection import RFE

rfe_accuracy_list = []
for k in range(1, 14):
   RFE_selector = RFE(estimator=gbc, n_features_to_select=k, step=1)
   RFE_selector.fit(x_tarin, y_train)

   selected_feature_x_train = RFE_selector.transform(x_tarin)
   selected_feature_x_test = RFE_selector.transform(x_test)

   gbc.fit(selected_feature_x_train, y_train)
   RFE_preds = gbc.predict(selected_feature_x_test)

   accuracy_rfe = round(accuracy_score(y_test, RFE_preds), 4)
   rfe_accuracy_list.append(accuracy_rfe)
   # acc_p(k)

print("RFE Accuracy:",rfe_accuracy_list)

"Ploting Bar graph"
fig, ax = plt.subplots()

x = np.arange(1, 14)
y = rfe_accuracy_list

ax.bar(x, y, width=0.2)
ax.set_xlabel("Number of features selected using RFE")
ax.set_ylabel("Accuracy of model")
ax.set_ylim(0, 1.2)
ax.set_xticks(np.arange(1, 14))
ax.set_xticklabels(np.arange(1, 14), fontsize=12)

for idx, val in enumerate(y):
   plt.text(x=idx + 1, y=val + 0.001, s=str(val), ha="center")

plt.tight_layout()



RFE_selector = RFE(estimator=gbc, n_features_to_select=3, step=1)
RFE_selector.fit(x_tarin, y_train)

selected_feature_mask = RFE_selector.get_support()
selected_features = x_tarin.columns[selected_feature_mask]
print(f"selected features: {selected_features}")