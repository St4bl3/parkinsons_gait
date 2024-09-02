from sklearn.neural_network import MLPClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

dl_clf = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 20, 10), solver='adam')
dl_clf = evaluate_classifier(dl_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(dl_clf, X_train, y_train)

# Save the model
joblib.dump(dl_clf, 'deep_learning_model.pkl')
