import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from matplotlib import style

dataset1 = pd.read_csv('Resources/Student_mat.csv')

data = dataset1[['G1','G2','G3','studytime','failures','absences']]

predict = 'G3'

X = np.array(data.drop([predict],1))
y = np.array(data[predict])

X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(X_train,y_train)
accuracy = linear.score(X_test,y_test)

print(accuracy)
print("Coefficient = ",linear.coef_)
print("Intercept = ",linear.intercept_)

predictions = linear.predict(X_test)
for x in range(len(predictions)):
    print(predictions[x],X_test[x],y_test[x])


p = 'G1'
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()