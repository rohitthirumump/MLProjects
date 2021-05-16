# Importing The Libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.ensemble  import RandomForestClassifier


#Importing The Dataset

dataset = pd.read_csv("Resources/winequality-red.csv")

dataset.head()

dataset.info()

dataset.describe()

# Check if There is any Null values
dataset.isnull().sum()

# Pair Plot
sb.pairplot(dataset)
plt.show()

#Drawing Histogram of the Dataset
dataset.hist(bins=20,figsize=(10,10))
plt.show()

# Drawing Heatmap of dataset
plt.figure(figsize=[18,7])
sb.heatmap(dataset.corr(),annot=True)
plt.show()

fig, ax = plt.subplots(ncols = 6, nrows = 2, figsize = (20, 10))
index = 0
ax = ax.flatten()

# Showing Boxplot of dataset
for col, value in dataset.items():
  sb.boxplot(y = col, data = dataset, ax = ax[index])
  index += 1

plt.tight_layout(pad = 0.5, w_pad = 0.7, h_pad = 5.0)

fig, ax = plt.subplots(ncols = 6, nrows = 2, figsize = (20, 10))
index = 0
ax = ax.flatten()

for col, value in dataset.items():
  sb.distplot(value, ax = ax[index])
  index += 1

plt.tight_layout(pad = 0.5, w_pad = 0.7, h_pad = 5.0)

# Changing quality column into good and bad
dataset['quality'] = ['Good' if x>=7 else 'bad' for x in dataset['quality']]
dataset.head()

# Splitting the dataset into X and y
X = dataset.drop(columns = ['quality'])
y = dataset['quality']

y.value_counts()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

# Predicting The quality of wine by user input
def Prediction(model):
  # Predicting the value
  index = dataset.columns.values
  X_prediction = []
  for i in range(0,11):
    a = float(input("Enter {} = ".format(index[i])))
    X_prediction.append(a)
  prediction = model.predict([X_prediction])
  print('Prediction = ', prediction)
  return prediction[0]


def Model(X,y,model):
  #Traning the model
  model = model.fit(X, y)

  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

  print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
  return model


# KNN
model = KNeighborsClassifier()
Model(X,y,model)
Prediction(model)

#Random Forest
model = RandomForestClassifier()
Model(X,y,model)
Prediction(model)

# SVM
model = SVC()
Model(X,y,model)
Prediction(model)
