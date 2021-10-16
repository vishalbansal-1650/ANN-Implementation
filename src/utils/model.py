import os
import io
import time
import logging

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf



def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28],name='inputlayer1'),
        tf.keras.layers.Dense(units=300, activation='relu', name='hiddenlayer1'),
        tf.keras.layers.Dense(units=100, activation='relu', name='hiddenlayer2'),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', name='outputlayer')
    ]

    logging.info("Defining the model architecture")

    def _get_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str


    model_clf = tf.keras.models.Sequential(layers=LAYERS)

    logging.info(f"Model architecture: \n{_get_model_summary(model_clf)}")

    logging.info("Compiling the model")
    model_clf.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    logging.info("Model Compiled successfully")

    return model_clf ## <<< untrained model


def getCallbacks(logpath,CKPT_path)-> list:

    ## defining tensorboard callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logpath)

    ## defining early stopping callbacks
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    ## defining model checkpointing callbacks

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]

    return CALLBACKS_LIST


def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def save_model(model,model_name,model_dir):
    logging.info("Saving the trained model")
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir,unique_filename)
    model.save(path_to_model)
    logging.info(f"Trained model saved at path : {path_to_model}")


def save_plot(model_history, plot_name, plot_dir):
    logging.info("Saving the model performance plot")
    df = pd.DataFrame(model_history.history)

    df.plot(figsize=(10,8))
    plt.grid(True)
    unique_filename = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_filename)

    plt.savefig(path_to_plot)
    logging.info(f"Saved plot at path : {path_to_plot}")
