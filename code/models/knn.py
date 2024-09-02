from sklearn.neighbors import KNeighborsClassifier
import joblib
from utils import load_data, evaluate_classifier, plot_learning_curves

X_train, X_test, y_train, y_test = load_data('D:\\B-TECH\\SEM_4\\machine_learning\\final\\dataset\\final.csv')

knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=5, n_neighbors=3, weights='uniform')
knn_clf = evaluate_classifier(knn_clf, X_train, X_test, y_train, y_test)
plot_learning_curves(knn_clf, X_train, y_train)

# Save the model
joblib.dump(knn_clf, 'knn_model.pkl')
