from sklearn.ensemble import AdaBoostClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

ada_clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
ada_clf = evaluate_classifier(ada_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(ada_clf, X_train, y_train)

# Save the model
joblib.dump(ada_clf, 'adaboost_model.pkl')
