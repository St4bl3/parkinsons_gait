from sklearn.naive_bayes import GaussianNB
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

nb_clf = GaussianNB()
nb_clf = evaluate_classifier(nb_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(nb_clf, X_train, y_train)

# Save the model
joblib.dump(nb_clf, 'naive_bayes_model.pkl')
