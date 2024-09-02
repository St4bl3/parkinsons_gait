from sklearn.linear_model import LogisticRegression
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves, scale_data

X_train, X_test, y_train, y_test = load_data(r'AD:\B-TECH\SEM_4\machine_learning\final\dataset\final.csv')
X_train, X_test = scale_data(X_train, X_test)

lr_clf = LogisticRegression(class_weight='balanced', fit_intercept=True, solver='sag', max_iter=1000)
lr_clf = evaluate_classifier(lr_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(lr_clf, X_train, y_train)

# Save the model
joblib.dump(lr_clf, 'logistic_regression_model.pkl')
