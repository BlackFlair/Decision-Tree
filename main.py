import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

balance_data = pd.read_csv('', sep=',', header=0)

balance_data.head()

x = balance_data.values[:, 1:5]
y = balance_data.values[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train,y_train)

y_pred_en = clf_entropy.predict(x_test)

acc = accuracy_score(y_test, y_pred_en)
print("Accuracy : ", acc*100, "%")