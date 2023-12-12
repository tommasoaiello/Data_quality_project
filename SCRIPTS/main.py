import pandas as pd
import numpy as np
from A_data_collection import make_dataset_for_classification
from D_data_analysis import classification
from E_plot_results import plot
from generate_correlated_features import inject_correlated_feature
# DEFAULT PARAMETERS FOR CLASSIFICATION, REGRESSION & CLUSTERING
# N.B. CAN BE CHANGED ACCORDING TO THE ASSIGNMENT GUIDELINES & THE DATA QUALITY ISSUE TO BE INJECTED
#X, y = make_dataset_for_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, seed=2023)
#X, y = make_dataset_for_regression(n_samples=1000, n_features=3, n_informative=3, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, seed=2023)
#X = make_dataset_for_clustering(n_samples=1000, n_features=4, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), seed=2023)

# LIST OF ALGORITHMS FOR CLASSIFICATION, REGRESSION & CLUSTERING
CLASSIFICATION_ALGORITHMS = ["DecisionTree","LogisticRegression","KNN","RandomForest","AdaBoost","MLP"]
REGRESSION_ALGORITHMS = ["LinearRegressor","BayesianRidge","GPRegressor","SVMRegressor","KNNRegressor","MLPRegressor"]
CLUSTERING_ALGORITHMS = ["KMeans","Agglomerative","Spectral","OPTICS","BIRCH"]

SEED = 2023

if __name__ == '__main__':

    print("Main ...")
    

    #print("Experiment 1: One more feature, increasing correlation")
    # A: DATA COLLECTION
    # X, y = make_dataset_for_classification(+ parameters)
    

    # B: DATA POLLUTION
    # YOUR POLLUTION FUNCTION (on the generated datasets)

    # D: DATA ANALYSIS
    #for algorithm in CLASSIFICATION/REGRESSION/CLUSTERING_ALGORITHMS:
        #results_1_analysis = classification(X, y, algorithm, SEED)
        #results_1_analysis = regression(X, y, algorithm, SEED)
        #results_1_analysis = clustering(X, algorithm, n_clusters, SEED)

    # C: DATA PREPARATION (IF REQUESTED, IT DEPENDS ON THE DQ ISSUE)
    # YOUR DATA PREPARATION FUNCTION (to apply on the generated datasets)

    # D: DATA ANALYSIS ON THE CLEANED DATASETS (re-compute the data analysis evaluation on prepared dataset)
    #for algorithm in CLASSIFICATION/REGRESSION/CLUSTERING_ALGORITHMS:
        #results_2_analysis = classification(X, y, algorithm, SEED):
        #results_2_analysis = regression(X, y, algorithm, SEED)
        #results_2_analysis = clustering(X, algorithm, n_clusters, SEED)

    # E: PLOT RESULTS
    # N.B. IF YOU WANT TO SAVE THE PLOT OR CHANGE THE y_lim THE INSTRUCTIONS ARE INSIDE THE E_plot results
    # plot(x_axis_values, x_label, results, title, algorithms, plot_type)
    # WHERE plot_type = "performance" ("distance train-test" only for classification/regression) "speed"
    # ADD TABLES WITH THE RESULTS!








    # ------------------------------------------------------------------------------------------------------

    # CLASSIFICATION EXAMPLE
    # (in this toy example, I simply created 10 datasets with respectively 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009 number of samples
    # and I generated the three plots for the evaluation of the results)
    results_for_each_algorithm = []
    for algorithm in CLASSIFICATION_ALGORITHMS: # FIRST CICLE ON THE ALGORITHMS

        print('qui')
        results_single_algorithm = []

        correlation_strength = 0.5

        X, y = make_dataset_for_classification(n_samples=1000, n_features=7, n_informative=7, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, seed=2023)
        results_base_analysis = classification(X, y, algorithm, SEED)
        results_single_algorithm.append(results_base_analysis)

        for i in range(0, 6): # SECOND CICLE ON THE NUMBER OF POLLUTED DATASET THAT YOU WANT TO CREATE WITH DIFFERENT % OF POLLUTION
            # DATA COLLECTION: this is going to be eliminated
            #X, y = make_dataset_for_classification(n_samples=1000, n_features=7, n_informative=7, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, seed=2023)

            #Data pollution
            
            existing_feature_index = 0  # Choose the existing feature with which you want to correlate
            correlation_strength = correlation_strength + i*0.1  # Adjust the correlation strength
            
            X_with_new_feature = inject_correlated_feature(X, existing_feature_index, correlation_strength)
            # DATA ANALYSIS
            

            results_1_analysis = classification(X_with_new_feature, y, algorithm, SEED)
            results_single_algorithm.append(results_1_analysis)

        results_for_each_algorithm.append(results_single_algorithm)

    # RESULTS EVALUATION
    #EXAMPLE FOR USING THE SCRIPTS TO PLOT THE RESULTS
    plot(x_axis_values=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], x_label="Correlation values", results=results_for_each_algorithm, title="1 Plot trial classification perf", algorithms=CLASSIFICATION_ALGORITHMS, plot_type="performance")
    plot(x_axis_values=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], x_label="Correlation values", results=results_for_each_algorithm, title="2 Plot trial classification dist", algorithms=CLASSIFICATION_ALGORITHMS, plot_type="distance train-test")
    plot(x_axis_values=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], x_label="Correlation values", results=results_for_each_algorithm, title="3 Plot trial classification speed", algorithms=CLASSIFICATION_ALGORITHMS, plot_type="speed")





