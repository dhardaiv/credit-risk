

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv('credit_risk_dataset.csv')
df

df.head()

df.tail()

#begin data cleaning

df.shape

print(f'rows: {df.shape[0]}')
print(f'cols: {df.shape[1]}')
df.info()

#checking null values
df.isnull().sum()

#unique values in each col
df.nunique()

#dropping the NaN values
df.dropna(axis=0,inplace=True)

df.isnull().sum()

df.shape

#identifying outliers

df.dtypes

num_cols= df.select_dtypes(include=['int64', 'float64'])
num_cols

##choosing outlier as age>80 and checking count:
outlier_age= df[df["person_age"] > 80].shape[0]
outlier_age

#dropping count in shape[0] when age>80
df[df["person_age"] > 80].shape[0]
df.shape

sns.histplot(data= df, x='person_age')
plt.show()

"""We can see that count drastically decreases when age>60. so, removing count for age>80 was a good choice. Now, we can begin building our model - first beginning with splitting the training and test data."""

from sklearn.model_selection import train_test_split
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import necessary libraries for the mentioned models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    'XGBoost': XGBClassifier()
}

model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    model_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }

  # Convert results to DataFrame
results_df = pd.DataFrame(model_results).T
print(results_df)
