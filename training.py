from pathlib import Path

import joblib
from keras.applications import vgg19
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from constants import EPOCHS

data_path = Path("data")

x_train, y_train = (
    joblib.load(data_path / "x_train.joblib"),
    joblib.load(data_path / "y_train.joblib"),
)
x_test, y_test = (
    joblib.load(data_path / "x_test.joblib"),
    joblib.load(data_path / "y_test.joblib"),
)

x_train = vgg19.preprocess_input(x_train)
pretrained_nn = vgg19.VGG19(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
x_train_features = pretrained_nn.predict(x_train)
x_test_features = pretrained_nn.predict(x_test)

y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

joblib.dump(x_train_features, data_path / "x_train_features.joblib")
joblib.dump(x_test_features, data_path / "x_test_features.joblib")

# Create a model and add layers
model = Sequential()
model.add(Flatten(input_shape=x_train_features.shape[1:]))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))
# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath="data/model.best.hdf5", verbose=1, save_best_only=True)
# Train the model
model.fit(
    x_train_features,
    y_train_cat,
    callbacks=[early_stopping, checkpointer],
    epochs=EPOCHS,
    shuffle=True,
    validation_split=0.1,
)
print(model.evaluate(x_test_features, y_test_cat))

print("Model Summary")
print(model.summary())
# Save neural network structure
model_structure = model.to_json()
f = data_path / "model_structure.json"
f.write_text(model_structure)
