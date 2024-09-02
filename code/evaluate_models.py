import joblib
from models.utils import load_data, evaluate_classifier, scale_data

# Load data
X_train, X_test, y_train, y_test = load_data(r'D:\B-TECH\SEM_4\machine_learning\final\dataset\final.csv')

# Scale the data
X_train, X_test = scale_data(X_train, X_test)

# List of models to evaluate with updated paths
models = [
    ('Decision Tree', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\decision_tree_model.pkl'),
    ('Perceptron', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\perceptron_model.pkl'),
    ('Deep Learning', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\deep_learning_model.pkl'),
    ('SVM', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\svm_model.pkl'),
    ('Naive Bayes', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\naive_bayes_model.pkl'),
    ('Logistic Regression', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\logistic_regression_model.pkl'),
    ('k-Nearest Neighbors', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\knn_model.pkl'),
    ('Bagging', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\bagging_model.pkl'),
    ('Random Forests', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\random_forest_model.pkl'),
    ('AdaBoost', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\adaboost_model.pkl'),
    ('Gradient Boosting', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\gradient_boosting_model.pkl'),
    ('CatBoost', r'D:\B-TECH\SEM_4\machine_learning\final\code\trained_models\catboost_model.pkl')
]

# Evaluate each model
for model_name, model_path in models:
    print(f'\nEvaluating {model_name}')
    clf = joblib.load(model_path)
    evaluate_classifier(clf, X_train, X_test, y_train, y_test)
