import datetime
import os

import dns
import numpy as np
import pymongo
import tensorflow as tf
from flask import (Flask, abort, jsonify, make_response, redirect,
                   render_template, request, url_for)
from flask_restplus import Api, Resource, fields
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from werkzeug.datastructures import FileStorage

graph = tf.get_default_graph()
application = app = Flask(__name__)

if os.environ.get("MUSER"):
    user = os.environ.get("MUSER")
    password = os.environ.get("PASSWORD")
    url = os.environ.get("URL")

    client = pymongo.MongoClient(f"mongodb+srv://{user}:{password}@{url}")
    db = client.logs
else:
    client = pymongo.MongoClient("mongodb://localhost/mnist")
    db = client.logs

api = Api(
    app,
    version="1.0",
    title="MNIST classification",
    description="Automated Number classification through Deep learning",
)
ns = api.namespace("MNIST", description="Methods")

#
single_parser = api.parser()
single_parser.add_argument(
    "file", type=FileStorage, required=True, help="Image to store", location="files"
)


app.model = load_model("mnist_cnn.h5")


@ns.route("/mnist")
class Mnist(Resource):
    """Uploads your data to the recommender system"""

    @api.doc(parser=single_parser, description="Enter an image")
    def post(self):
        """Uploads a new transaction to Rex (Click to see more)"""

        args = single_parser.parse_args()
        img: Image = Image.open(args.file)
        img = img.resize((28, 28))
        file = img_to_array(img)
        img.close()
        file = file.reshape(1, 1, 28, 28)
        file = file / 255

        with graph.as_default():
            all_pred = app.model.predict(file)
            pred = np.argmax(all_pred[0])

        request_info = {
            "pred": str(pred),
            "filename": args.file.filename,
            "time": datetime.datetime.now(),
        }

        db.logs.insert_one(request_info)

        return {"pred": str(pred)}


if __name__ == "__main__":
    app.run()
