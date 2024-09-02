from sklearn.neural_network import MLPClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves, scale_data

X_train, X_test, y_train, y_test = load_data(r'D:\B-TECH\SEM_4\machine_learning\final\dataset\final.csv')
X_train, X_test = scale_data(X_train, X_test)

nn_clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(30, 20, 15, 10), learning_rate='constant', solver='lbfgs', max_iter=1000)
nn_clf = evaluate_classifier(nn_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(nn_clf, X_train, y_train)

# Save the model
joblib.dump(nn_clf, 'neural_net_model.pkl')
