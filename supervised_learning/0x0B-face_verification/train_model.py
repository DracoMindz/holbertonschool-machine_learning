#!/usr/bin/env python3
"""
class TrainModel
trains model for face verification using triplet loss
"""

    def __init__(self, model_path, alpha):
        """
        class constructor
        :model_path: path to base face verification embedding model
                    loads model using
                    with tf.keras.utils.CustomObjectScope({'tf': tf}):
                    saves model as public instance method base_model
        :alpha: use for the triplet loss calculation
        :return: new model
        """
        # loads model and saves model
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        self.base_model.save(base_model)


