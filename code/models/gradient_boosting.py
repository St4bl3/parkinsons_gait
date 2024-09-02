from sklearn.ensemble import GradientBoostingClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

gb_clf = GradientBoostingClassifier(n_estimators=25)
gb_clf = evaluate_classifier(gb_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(gb_clf, X_train, y_train)

# Save the model
joblib.dump(gb_clf, 'gradient_boosting_model.pkl')
