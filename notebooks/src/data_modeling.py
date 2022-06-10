import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error


from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib


import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve

def split(data):
    y = data['TotalGHGEmissions']
    X = data.drop('TotalGHGEmissions', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
    return X, y, X_train, X_test, y_train, y_test

def col_transofrmer(): 
    numeric_transformer = Pipeline(steps=[
       ('scaler', MinMaxScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first',handle_unknown = 'ignore'))
    ])
    return numeric_transformer, categorical_transformer

def categories(X):
    cat = X.select_dtypes(include=["object"])
    non_cat = X.select_dtypes(exclude=["object"])
    return cat, non_cat


def categories_tolist(cat, non_cat):
    numeric_features = non_cat.columns.values.tolist()
    categorical_features = cat.columns.values.tolist()
    return numeric_features, categorical_features

def col_preprocessor(numeric_transformer, numeric_features, categorical_transformer, categorical_features):
    preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features)
    ,('categorical', categorical_transformer, categorical_features)
    ])
    return preprocessor

def preprocess_pipeline(preprocessor): 
    preprocess = Pipeline(steps=[('preprocessor', preprocessor)])
    return preprocess

def train_transform(preprocess, X_train):
    training_transformed = preprocess.fit_transform(X_train)
    pd.DataFrame(training_transformed)
    return training_transformed

def linear_reg(preprocessor, X_train, X_test, y_train, y_test):
    pipe_reglin = Pipeline(steps = [
                    ('preprocessor', preprocessor),
                    ('reglin', LinearRegression())
            ])
    pipe_reglin.fit(X_train,y_train)
    pipe_reglin.score(X_train,y_train)
    pipe_reglin_ypred = pipe_reglin.predict(X_test)
    return pipe_reglin, pipe_reglin_ypred

def random_forest(preprocessor, X_train, X_test, y_train, y_test):
    pipe_rfreg = Pipeline(steps = [
                    ('preprocessor', preprocessor),
                    ('regr', RandomForestRegressor(max_depth=2, random_state=0))
            ])
    pipe_rfreg.fit(X_train,y_train)
    pipe_rfreg.score(X_train,y_train)
    pipe_rfreg_ypred = pipe_rfreg.predict(X_test)
    return pipe_rfreg, pipe_rfreg_ypred

def show_results(y_test, pipeline_pred):
    print(f"MSE : {mean_squared_error(y_test, pipeline_pred)}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, pipeline_pred))}")
    print(f"MAE : {mean_absolute_error(y_test, pipeline_pred)}")

def pickle_export(pipe_rfreg):
    joblib.dump(pipe_rfreg, 'pipeline.pkl')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




def lc_test(data, X, y):
    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    
    # cv = ShuffleSplit(X.shape[0], n_iter=100, test_size=0.2, random_state=0)

    estimator = GaussianNB()
    plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=4)

    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    # cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10,
    #                                 test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.1)
    plot_learning_curve(estimator, title, X, y, cv=10, n_jobs=4)

    plt.show()