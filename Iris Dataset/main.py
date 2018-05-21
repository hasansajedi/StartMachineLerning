# Load libraries
from ml_data import ml_data_analyzer

mda = ml_data_analyzer("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                       ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])

mda.load_dataset()
mda.start_learn()
mda.start_test()
mda.select_best_model()
# mda.start_prediction("KNN")
# mda.start_prediction("SVC")
# mda.start_prediction("LoR")
# mda.start_prediction("LDA")
# mda.start_prediction("QDA")
# mda.start_prediction("GNB")
# mda.start_prediction("DT")
# mda.start_prediction("RF")
