import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves, scale_data

# Load data
X_train, X_test, y_train, y_test = load_data(r'D:\B-TECH\SEM_4\machine_learning\final\dataset\final.csv')

# Handle class imbalance manually
# Separate majority and minority classes
X = pd.concat([X_train, y_train], axis=1)
majority_class = X[X.iloc[:, -1] == X.iloc[:, -1].value_counts().idxmax()]
minority_class = X[X.iloc[:, -1] != X.iloc[:, -1].value_counts().idxmax()]

# Upsample minority class
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

# Combine majority class with upsampled minority class
upsampled = pd.concat([majority_class, minority_upsampled])

# Separate features and target variable from upsampled data
X_train = upsampled.iloc[:, :-1]
y_train = upsampled.iloc[:, -1]

# Scale the data
X_train, X_test = scale_data(X_train, X_test)

# Train the model
dt_clf = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_features=3)
dt_clf = evaluate_classifier(dt_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(dt_clf, X_train, y_train)

# Save the model
joblib.dump(dt_clf, 'decision_tree_model.pkl')
