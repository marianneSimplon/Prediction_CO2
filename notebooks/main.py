import pandas as pd
from src.data_processing import data_processing
from src.data_modeling import split, col_transofrmer, categories, categories_tolist, col_preprocessor, preprocess_pipeline, linear_reg, show_results, pickle_export, plot_learning_curve, lc_test, random_forest
import warnings
import os
import click
warnings.filterwarnings("ignore")



def run():
    print('Loading data...', end='')
    data = data_processing()
    print('ok')

    print('Performing split on data...', end='')
    X, y ,X_train, X_test, y_train, y_test = split(data)
    print('ok')


    print('Creating a pipeline and Encoding categorial features and Scaling numerical features...', end='')
    numeric_transformer, categorical_transformer = col_transofrmer()
    print('ok')

    print('Separating categorical from non-categorical features...', end='')
    cat, non_cat = categories(X)
    print('ok')

    print('Listing all features...', end='')
    numeric_features, categorical_features = categories_tolist(cat, non_cat)
    print('ok')

    print('Column transformer added to pipeline...', end='')
    preprocessor = col_preprocessor(numeric_transformer, numeric_features, categorical_transformer, categorical_features)
    print('ok')

    print('Pipeline running...', end='')
    preprocess = preprocess_pipeline(preprocessor)
    print('ok')

    print('Running Linear Regression through the pipeline...', end='')
    pipe_reglin, pipe_reglin_ypred = linear_reg(preprocessor, X_train, X_test, y_train, y_test)
    print('ok')

    print('Running Random Forest Regressor through the pipeline...', end='')
    pipe_rfreg, pipe_rfreg_ypred = random_forest(preprocessor, X_train, X_test, y_train, y_test)
    print('ok')

    print("Model runned, let's see the results")
    show_results(y_test, pipe_reglin_ypred)

    print("Model runned, let's see the results")
    show_results(y_test, pipe_rfreg_ypred)

    print('Random forest have better resutls')
    print("Saving model as pickle...", end='')
    pickle_export(pipe_rfreg)
    print('ok')


    lc_test(data, X, y)


if __name__ == "__main__":
    run()


