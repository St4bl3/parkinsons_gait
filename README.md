Parkinsons gait analysis. A video dataset with 87 videos with both healthy and pd. first take every consecutive frames and then take their euclidean distances from their X, and Y coordinates and store them in rows of csv file, making one csv file per video and every two frames have one row in csv file. Then after we get 87 csv files of dataset, we aggregate them into a single csv file, and then shuffel them. Then after shuffling we perform the following ml algorithms: adaboost, bagging, catboost, decision tree, deep learning, gradiant boostin, knn, logistic regression, naive bayes, neural net, perceptron, random forest, svm and find their corresponding accuracies and other performance metrics, and even their learning curves. and even utils.py file where all the trained models are called and again all the parameters are passed to get their accuracies and it is printed in the evaluate_models.py file. by the help of this we can predict the presence of parkinsons disease in any human, by processing the video of them walking..
