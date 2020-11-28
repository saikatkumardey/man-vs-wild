import matplotlib.pyplot as plt
import joblib
from constants import PEOPLE, WILD
import numpy as np

x_train = joblib.load("data/x_train.joblib")
y_train = joblib.load("data/y_train.joblib")

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))
for i in range(4):
    ax[i].imshow(x_train[np.where(y_train == PEOPLE)][i])
ax[0].set_ylabel("PEOPLE")
fig.savefig("data/class_people.png", bbox_inches="tight")


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))
for i in range(4):
    ax[i].imshow(x_train[np.where(y_train == WILD)][i])
ax[0].set_ylabel("WILD")
fig.savefig("data/class_wild.png", bbox_inches="tight")
