import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,r2_score, roc_auc_score
df = pd.read_csv("Fraud_Log.csv")
df.dropna(inplace=True)
df.head(5)
df.columns
df['type'].unique()
num_cols = ['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
numerics_df = df[num_cols].reset_index(drop=True)
df.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)
df.isnull().sum()
encoder_type = OneHotEncoder(sparse_output=False,drop = None)
type_en = encoder_type.fit_transform(df[['type']])
encoded_type = pd.DataFrame(type_en,columns=encoder_type.get_feature_names_out(['type']))
df_encoded = pd.concat([encoded_type,numerics_df],axis=1)
df_encoded.head(20)
type_c = df['type'].value_counts().reset_index()
type_c.columns = ['type','count']
plt.Figure(figsize=(10,5))
sns.barplot(x='type',y='count',hue='type',data = type_c,palette='Pastel1', legend=None)
plt.title('Transaction Types Plot')
plt.xlabel('Types of Transactions')
plt.ylabel('Transactions Count')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(type_c['count'], labels=type_c['type'], autopct='%1.1f%%', colors=sns.color_palette('PuBu'), startangle=140)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Transaction Types - Donut Chart')
plt.axis('equal')
plt.tight_layout()
plt.show()
x = df_encoded
y = df['isFraud'].values
scaled = StandardScaler()
x_scaler = scaled.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.7)
log = LogisticRegression(max_iter=100000,class_weight='balanced')
log.fit(x_train,y_train)
y_pred_log = log.predict(x_train)
y_pred_log_test = log.predict(x_test)
print("Logistic Train Accuracy: ", accuracy_score(y_train, y_pred_log)*100)
print("Logistic Test Accuracy: ", accuracy_score(y_test, y_pred_log_test)*100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_test))
print("Classification Report:\n", classification_report(y_test, y_pred_log_test))
dec = DecisionTreeClassifier(max_depth=10,min_samples_split=5,min_samples_leaf=4,class_weight='balanced')
dec.fit(x_train, y_train)
y_pred_train = dec.predict(x_train)
y_pred_dec = dec.predict(x_test)
print("Decision Tree Train Accuracy: ", accuracy_score(y_train, y_pred_train)*100)
print("Decision Tree Accuracy: ", accuracy_score(y_test, y_pred_dec)*100)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dec))
print("Classification Report:\n", classification_report(y_test, y_pred_dec))
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dec)
plt.title('Decision Tree - Confusion Matrix')
plt.show()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log_test)
plt.title('Logistic Regression - Confusion Matrix')
plt.show()
importances = dec.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=x.columns[indices])
plt.title('Feature Importance - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

