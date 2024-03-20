import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


irisdataset = pd.read_csv('Iris.csv')
irisdataset.drop('Id', axis=1, inplace=True)
irisdataset.columns = irisdataset.columns.str.strip()

X = irisdataset.drop('Species', axis=1) 
y = irisdataset['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
joblib.dump(model, 'iris.pkl')
