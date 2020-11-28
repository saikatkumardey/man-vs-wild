from pathlib import Path

import numpy as np
from keras.applications import vgg19
from keras.models import model_from_json
from keras.preprocessing import image
import click

# Load the json file that contains the model's structure
f = Path("data/model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("data/model.best.hdf5")


@click.command()
@click.option("--path", required=True, help="path to the image file to be classified")
def classify_image(path):
    img = image.load_img(path, target_size=(32, 32))
    # Convert the image to a numpy array
    image_array = image.img_to_array(img)
    # Add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
    images = np.expand_dims(image_array, axis=0)
    # Normalize the data
    images = vgg19.preprocess_input(images)
    # Use the pre-trained neural network to extract features from our test image (the same way we did to train the model)
    feature_extraction_model = vgg19.VGG19(
        weights="imagenet", include_top=False, input_shape=(32, 32, 3)
    )
    features = feature_extraction_model.predict(images)
    print(features.shape)
    prediction = np.round(model.predict_proba(features) * 100, 2)[0]
    pred_formatted = dict(zip(["WILD", "PEOPLE"], prediction))
    click.echo("Predicted probability (in %) => {}".format(pred_formatted))
    return prediction


if __name__ == "__main__":
    classify_image()
