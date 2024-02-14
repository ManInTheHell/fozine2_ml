import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

from conf import DATASET_PATH
from load_dataset import load_dataset


def linear():
    data, tag = load_dataset(DATASET_PATH)
    data_train, data_test, tag_train, tag_test = train_test_split(data, tag, test_size=0.2, random_state=10)

    model = LinearRegression()
    model.fit(data_train, tag_train)
    tag_predicted = model.predict(data_test)

    root_mse = np.sqrt(mean_squared_error(tag_test, tag_predicted))
    mae = mean_absolute_error(tag_test, tag_predicted)
    return mae, root_mse


def ridge():
    data, tag = load_dataset(DATASET_PATH)
    data_train, data_test, tag_train, tag_test = train_test_split(data, tag, test_size=0.2, random_state=10)

    model = Ridge(alpha=1.0)
    model.fit(data_train, tag_train)
    tag_predicted = model.predict(data_test)

    root_mse = np.sqrt(mean_squared_error(tag_test, tag_predicted))
    mae = mean_absolute_error(tag_test, tag_predicted)
    return mae, root_mse


def lasso():
    data, tag = load_dataset(DATASET_PATH)
    data_train, data_test, tag_train, tag_test = train_test_split(data, tag, test_size=0.2, random_state=10)

    model = Lasso(alpha=1.0)
    model.fit(data_train, tag_train)
    tag_predicted = model.predict(data_test)

    root_mse = np.sqrt(mean_squared_error(tag_test, tag_predicted))
    mae = mean_absolute_error(tag_test, tag_predicted)
    return mae, root_mse


def elastic_net():
    data, tag = load_dataset(DATASET_PATH)
    data_train, data_test, tag_train, tag_test = train_test_split(data, tag, test_size=0.2, random_state=10)

    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(data_train, tag_train)
    tag_predicted = model.predict(data_test)

    root_mse = np.sqrt(mean_squared_error(tag_test, tag_predicted))
    mae = mean_absolute_error(tag_test, tag_predicted)
    return mae, root_mse


def svr():
    data, tag = load_dataset(DATASET_PATH)
    data_train, data_test, tag_train, tag_test = train_test_split(data, tag, test_size=0.2, random_state=10)

    model = SVR(kernel='linear')
    model.fit(data_train, tag_train)
    tag_predicted = model.predict(data_test)

    root_mse = np.sqrt(mean_squared_error(tag_test, tag_predicted))
    mae = mean_absolute_error(tag_test, tag_predicted)
    return mae, root_mse
