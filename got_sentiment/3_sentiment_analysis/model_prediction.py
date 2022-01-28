
import os
import pickle
import numpy as np

class CustomModelPrediction(object):

  def __init__(self, model, processor):
    # Class gets instantiated with a trained model file and a persisted processor
    self._model = model
    self._processor = processor

  def _postprocess(self, predictions):
    # Create an output signature
    labels = ['negative', 'positive']
    return [
        {
            "label":labels[int(np.round(prediction))],
            "score":float(np.round(prediction,4))
        } for prediction in predictions]

  def predict(self, instances, **kwargs):
    # Clean the data, make predictions and postprocess
    preprocessed_data = self._processor.transform(instances)
    predictions =  self._model.predict(preprocessed_data)
    labels = self._postprocess(predictions)
    return labels

  @classmethod
  def from_path(cls, model_dir):
    # Load the keras model and the persisted processor
    import tensorflow.keras as keras
    
    model = keras.models.load_model(
      os.path.join(model_dir,'keras_saved_model.h5'))
    
    # I know, pickle is bad and I should feel bad
    with open(os.path.join(model_dir, 'processor_state.pkl'), 'rb') as f:
      processor = pickle.load(f)

    return cls(model, processor)
