"""
 - Use a mease of importance to select the top k features
 - k is a subjective number eg:- top 3, 5, ...
   |-------------------|------------------------|-----------------------------|
   | target \ features | Categorical            |    Numerical                |
   |-------------------|------------------------|-----------------------------|
   |                   | Chi squared            | t-test                      |
   | Categorical       | Mutual information     | Mutual information          |
   |                   |                        |                             |
   |-------------------|------------------------|-----------------------------|
   | Numerical         | t-test                 | Pearson correlation         |
   |                   | Mutual information     | Spearman's rank correlation |
   |                   |                        | Mutual information          |
   |-------------------|------------------------|-----------------------------|
"""


import pandas  as pd 
import numpy as np
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
from sklearn.metrics import accuracy_score

# Initialize classifier
gbc = GradientBoostingClassifier(max_depth=5, random_state=42)

# Train Classifier using all features
gbc.fit(x_train, y_tarin)

# Make predictions
preds = gbc.predict(x_test)

# Evaluate the model using the f1-score
accuracy = round(accuracy_score(y_test, preds), 4)
print(f"Accuracy Score: {accuracy}")


# K-best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

"To filnd which feature gives best accuracy"
accuracy_list = []
for k in range(1, 14):
   selector = SelectKBest(mutual_info_classif, k=k)
   selector.fit(x_train, y_tarin)

   selected_feature_x_train = selector.transform(x_train) 
   selected_feature_x_test = selector.transform(x_test) 

   gbc.fit(selected_feature_x_train, y_tarin)
   k_best_preds = gbc.predict(selected_feature_x_test)

   accuracy_kbest = round(accuracy_score(y_test, k_best_preds) ,4)
   accuracy_list.append(accuracy_kbest)
print(accuracy_list)

# '''
fig, ax = plt.subplots()
x=np.arange(1, 14)
y=accuracy_list
ax.plot(x, y)
ax.set_title('Filter Method')
ax.bar(x, y, width=0.2)
ax.set_xlabel('No of features selected using mutual information')
ax.set_ylabel('Accuracy score')
ax.set_xticks(0, 1.2)
ax.set_xticklabels(np.arange(1, 14))

for index, value in enumerate(y):
   plt.text(x=int(index), y=value+1, s=str(value), ha='center')
plt.tight_layout()
# '''

"""
Analysis:-
 - here accuracy is [0.7593, 0.9259, 0.9815, 0.9815, 0.9815, 0.9815, 0.9815, 0.963, 0.963, 0.963, 0.9444, 0.9074, 0.9074]
 - as you can see if we select even only 3 or 4 feature we can get better performance than seecting 7 or 8,10,13...
 - from this we can reduce the complexicity of models
"""

selector = SelectKBest(mutual_info_classif, k=4)
selector.fit(x_train, y_tarin)

selected_feature_mask = selector.get_support()
selected_features = x_train.columns[selected_feature_mask]

print(selected_features)