from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), bootstrap=True, n_estimators=20)
bag_clf = evaluate_classifier(bag_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(bag_clf, X_train, y_train)

# Save the model
joblib.dump(bag_clf, 'bagging_model.pkl')
