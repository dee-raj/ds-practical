import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_wine
# %matplotlib inline
plt.rcParams["figure.figsize"] = (12, 9)

# READ THE DATA
wine_data = load_wine()
wine_df = pd.DataFrame(
   data=wine_data.data,
   columns=wine_data.feature_names
)
wine_df['target'] = wine_data.target
# print(wine_df)

'''
from seaborn import swarmplot
data_to_plot = pd.melt(wine_df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash','target']],
                       id_vars='target',
                       var_name='features',
                       value_name='value'
                     )
swarmplot(data=data_to_plot, x='features', y='value', hue='target')
'''

'freaquency of each classes'
tot_targ = wine_df['target'].value_counts()
print(tot_targ)
'''
fig, ax = plt.subplots()
x=[0, 1, 2]
y=[59, 71, 48]
ax.plot(x, y)
ax.set_title('Simple plot')
ax.bar(x, y, width=0.2)
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([0, 1, 2], fontsize=12)

for index, value in enumerate(y):
   plt.text(x=int(index), y=value+1, s=str(value), ha='center')
plt.tight_layout()
'''


"Train / Test split"
# '''
from sklearn.model_selection import train_test_split

X= wine_df.drop(['target'], axis=1)
Y=wine_df['target']

x_tarin, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=Y,
                                                    random_state=42
                                                   )

# print(x_tarin.shape)
# print(x_test.shape)
# '''



"Baseline Model: Gradient Boosting Classifier with all features"
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

# Initialize classifier
gbc = GradientBoostingClassifier(max_depth=5, random_state=42)

# Train Classifier using all features
gbc.fit(x_tarin, y_train)

# Make predictions
preds = gbc.predict(x_test)

# Evaluate the model using the f1-score
f1_score_all = round(f1_score(y_test, preds, average='weighted'), 3)
print(f"F1 Score: {f1_score_all}")