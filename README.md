# 🌸 Iris Flower Classifier with KNN

This project uses **scikit-learn** and the classic **Iris dataset** to classify flowers into three species: `setosa`, `versicolor`, and `virginica`.

## 📦 Libraries
- `sklearn.datasets`
- `KNeighborsClassifier` (KNN)
- `train_test_split`
- `accuracy_score`
- `matplotlib.pyplot` for visualization

## 🧠 Model
- Algorithm: **K-Nearest Neighbors (KNN)** with `k=3`
- Data split: 80% training, 20% testing
- Accuracy calculated using `score()` and `accuracy_score()`

## 📊 Dataset
- Built-in Iris dataset from `sklearn.datasets`
- Features: sepal length/width, petal length/width
- Labels: flower species (0 = setosa, 1 = versicolor, 2 = virginica)

## 📈 Example Plot
Simple line and scatter plot added for illustration using `matplotlib`.

## ✅ Output
- Predictions for test samples
- Final classification accuracy printed to the console
