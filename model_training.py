import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

#importing dataset
df= pd.read_csv('data/creditcard.csv')
print(df['Class'].value_counts())
X=df.drop('Class', axis=1)
y=df['Class']
class_counts = y.value_counts()

#Preprocessing
plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'orange'])
plt.xticks([0, 1], ['Legitimate (0)', 'Fraud (1)'])
plt.yscale('log')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count(Log Scale)')
plt.show()

#Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)
print("Training Class Distribution:\n", y_train.value_counts())
print("Testing Class Distribution:\n", y_test.value_counts())

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("Scaled Training Features:\n", X_train_scaled[:5])

#Transform
X_test_scaled = scaler.transform(X_test)
print("Scaled Test Features post transform:\n", X_test_scaled[:5])

#Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Classification Report for logistic regression model:\n", classification_report(y_test, y_pred))

#Tuning
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
model_rf.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Classification Report for random forest model:\n", classification_report(y_test, y_pred))

#evaluation and adjustments
y_prob = model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.3
y_pred_adjusted = (y_prob > threshold).astype(int)
print("Classification Report with Adjusted Threshold:\n", classification_report(y_test, y_pred_adjusted))

joblib.dump(model, 'models/logreg_fraud_detection_model.pkl')
joblib.dump(model_rf, 'models/rf_fraud_detection_model.pkl')
print("Models saved successfully!")