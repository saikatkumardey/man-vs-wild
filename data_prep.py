from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar100
from constants import PEOPLE, WILD, CLASS_LABELS, PEOPLE_LABELS, WILD_LABELS

data_path = Path("data")

# load dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

class_labels_to_idx = {class_label.decode(): idx for idx, class_label in enumerate(CLASS_LABELS)}

people_label_idx = [class_labels_to_idx[label] for label in PEOPLE_LABELS]
wild_labels_idx = [class_labels_to_idx[label] for label in WILD_LABELS]

y_train_wild_condition = np.where(np.isin(y_train, wild_labels_idx) == True)[0]
y_test_wild_condition = np.where(np.isin(y_test, wild_labels_idx) == True)[0]

x_train_wild = x_train[y_train_wild_condition]
y_train_wild = np.array([WILD] * x_train_wild.shape[0])

x_test_wild = x_test[y_test_wild_condition]
y_test_wild = np.array([WILD] * x_test_wild.shape[0])

y_train_people_condition = np.where(np.isin(y_train, people_label_idx) == True)[0]
y_test_people_condition = np.where(np.isin(y_test, people_label_idx) == True)[0]

x_train_people = x_train[y_train_people_condition]
y_train_people = np.array([PEOPLE] * x_train_people.shape[0])

x_test_people = x_test[y_test_people_condition]
y_test_people = np.array([PEOPLE] * x_test_people.shape[0])


# combine training data
x_train_combined = np.concatenate([x_train_people, x_train_wild])
y_train_combined = np.concatenate([y_train_people, y_train_wild])

# combine test data
x_test_combined = np.concatenate([x_test_people, x_test_wild])
y_test_combined = np.concatenate([y_test_people, y_test_wild])


joblib.dump(x_train_combined, data_path / "x_train.joblib")
joblib.dump(y_train_combined, data_path / "y_train.joblib")
joblib.dump(x_test_combined, data_path / "x_test.joblib")
joblib.dump(y_test_combined, data_path / "y_test.joblib")


# view PEOPLE images
plt.imshow(x_train_people[3])
plt.title("People")


# view Wild images
plt.imshow(x_train_wild[5])
plt.title("Wild")
