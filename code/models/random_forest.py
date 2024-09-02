from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

rf_clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_features=2, n_estimators=5)
rf_clf = evaluate_classifier(rf_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(rf_clf, X_train, y_train)

# Save the model
joblib.dump(rf_clf, 'random_forest_model.pkl')
