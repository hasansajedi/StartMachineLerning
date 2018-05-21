# Load libraries
import logging
import pandas
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ml_data_analyzer():
    logger = None

    def __init__(self, file_url, file_cols):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Start main function')

        self.url = file_url
        self.names = file_cols

        self.dataSet = None
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = None, None, None, None
        self.results = []

        self.models = []
        self.selected_model = None
        self.model_names = []
        self.stds, self.means = [], []

        self.logger.info('Finish main function')

        # self.load_dataset()

    def load_dataset(self):
        # Load dataset
        self.dataSet = pandas.read_csv(self.url, names=self.names)

        # 1. Dimensions of the dataset.                    # shape
        self.logger.debug('shape: %s', self.dataSet.shape)

        # 2. Peek at the data itself.                      # head
        self.logger.debug('head: %s', self.dataSet.head(20))

        # 3. Statistical summary of all attributes.        # descriptions
        self.logger.debug('descriptions: %s', self.dataSet.describe())

        # 4. Breakdown of the data by the class variable.  # class distribution
        self.logger.debug('class distribution: %s', self.dataSet.groupby('class').size())

    def start_learn(self):
        # Split-out validation dataset
        array = self.dataSet.values
        X = array[:, 0:4]
        Y = array[:, 4]
        validation_size = 0.20
        seed = 7
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(X, Y,
                                                                                                            test_size=validation_size,
                                                                                                            random_state=seed)

    def start_test(self):
        # Test options and evaluation metric

        seed = 7
        scoring = 'accuracy'

        '''
        Let’s evaluate 6 different algorithms:
        
        Logistic Regression (LR)
        Linear Discriminant Analysis (LDA)
        K-Nearest Neighbors (KNN).
        Classification and Regression Trees (CART).
        Gaussian Naive Bayes (NB).
        Support Vector Machines (SVM).
        '''

        # Spot Check Algorithms
        self.models = []
        self.models.append(("SVC", SVC()))
        self.models.append(("KNN", KNeighborsClassifier()))
        self.models.append(("LoR", LogisticRegression()))
        self.models.append(("LDA", LinearDiscriminantAnalysis()))
        self.models.append(("QDA", QuadraticDiscriminantAnalysis()))
        self.models.append(("GNB", GaussianNB()))
        self.models.append(("DT", DecisionTreeClassifier()))
        self.models.append(("RF", RandomForestClassifier()))
        self.results = []
        self.names = []
        self.model_names = []

        for name, model in self.models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold, scoring=scoring)

            # print("\n" + name)
            self.model_names.append(name)
            # print("Result: " + str(cv_results))
            # print("Mean: " + str(cv_results.mean()))
            # print("Standard Deviation: " + str(cv_results.std()))
            self.means.append(cv_results.mean())
            self.stds.append(cv_results.std())

    def select_best_model(self):
        # Select Best Model
        x_loc = np.arange(len(self.models))
        width = 0.5
        models_graph = plt.bar(x_loc, self.means, width, yerr=self.stds)
        plt.ylabel('Accuracy')
        plt.title('Scores by models')
        plt.xticks(x_loc, self.model_names)  # models name on x-axis

        # add valve on the top of every bar
        def addLabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%f' % height, ha='center', va='bottom')

                addLabel(models_graph)

        self.logger.info('Best model is (%s) and model score is (%s)', self.models[self.stds.index(min(self.stds))][0],
                         min(self.stds))
        self.start_prediction(self.models[self.stds.index(min(self.stds))][0])

        plt.show()

        # Compare Algorithms
        # fig = plt.figure()
        # fig.suptitle('Algorithm Comparison')
        # ax = fig.add_subplot(111)
        # plt.boxplot(self.results)
        # ax.set_xticklabels(self.names)
        # plt.show()

    def start_prediction(self, type):
        # 6. Make Predictions    --   Make predictions on validation dataset
        model = None
        if type == "KNN":
            model = KNeighborsClassifier()
        elif type == "SVC":
            model = SVC()
        elif type == "LoR":
            model = LogisticRegression()
        elif type == "LDA":
            model = LinearDiscriminantAnalysis()
        elif type == "QDA":
            model = QuadraticDiscriminantAnalysis()
        elif type == "GNB":
            model = GaussianNB()
        elif type == "DT":
            model = DecisionTreeClassifier()
        elif type == "RF":
            model = RandomForestClassifier()

        model.fit(self.X_train, self.Y_train)
        predictions = model.predict(self.X_validation)
        self.logger.info(type + ' score is: %f%%', (accuracy_score(self.Y_validation, predictions)) * 100)
        self.logger.info('Confusion matrix is: %s', confusion_matrix(self.Y_validation, predictions))
        self.logger.info('Classification report is: %s', classification_report(self.Y_validation, predictions))
