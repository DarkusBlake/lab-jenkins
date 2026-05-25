import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Загружаем датасет Iris
iris = load_iris()

# Превращаем в DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Сохраняем в CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data downloaded and saved successfully!")
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
