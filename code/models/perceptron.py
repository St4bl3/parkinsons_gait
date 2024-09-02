from sklearn.linear_model import Perceptron
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

perc_clf = Perceptron(class_weight='balanced', fit_intercept=True, max_iter=90, penalty='l1', shuffle=True)
perc_clf = evaluate_classifier(perc_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(perc_clf, X_train, y_train)

# Save the model
joblib.dump(perc_clf, 'perceptron_model.pkl')
