from sklearn import datasets

from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score

y_true =[0,1,2,0,2,1,0,2,5,8]
y_pred=[0,1,1,0,2,1,0,0,8,8]
accuracy_score(y_true,y_pred)

import matplotlib.pyplot as plt
x=[0,2,4,6,8]
y=[1,3,5,7,9]
plt.plot(x,y)
plt.title("My Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x,y,label="points",color="red",marker="o")
plt.show()

iris=datasets.load_iris()
x=iris.data
y=iris.target
print(iris.target_names) #{'setosa', 'versicolor', 'virginica'}: three species of iris flowers
print(iris.feature_names)
print(x[0])
print(y[0])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

a=[0,2,5,1,8]
b=[0,1,0,1,1]
print(a_train)
print(a_test)

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

print(knn.predict([x_test[0]]))
print(y_test[0])
print(knn.predict([x_test[5]]))
print(y_test[5])
print(knn.predict([x_test[10]]))
print(y_test[10])
y_pred=knn.predict(x_test)
print(y_pred)
print(y_test)

accuracy=knn.score(x_test,y_test)
print(accuracy)
