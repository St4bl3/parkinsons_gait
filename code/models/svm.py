from sklearn.svm import SVC
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

svm_clf = SVC(class_weight=None, degree=3, shrinking=False)
svm_clf = evaluate_classifier(svm_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(svm_clf, X_train, y_train)

# Save the model
joblib.dump(svm_clf, 'svm_model.pkl')
