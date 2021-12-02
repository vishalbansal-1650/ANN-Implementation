import tensorflow as tf
import os
import numpy as np
import time

from utils.common import get_timestamp

def getCallbacks(config,x_train):

    logs_dir = config["logs"]["logs_dir"]
    unique_dir_name = get_timestamp("tb_logs")

    tensorboard_logs = config["logs"]["tensorboard_logs"]
    tb_log_path = os.path.join(logs_dir,tensorboard_logs,unique_dir_name)
    os.makedirs(tb_log_path,exist_ok=True)

    ## defining tensorboard callbacks
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_path)

    file_writer = tf.summary.create_file_writer(logdir=tb_log_path)

    with file_writer.as_default():
        images = np.reshape(x_train[10:30], (-1,28,28,1))
        tf.summary.image("20 Hand written digit samples", images, max_outputs=25, step=0)

    ## defining early stopping callbacks

    params = config["params"]

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience= params["patience"], 
        restore_best_weights= params["restore_best_weights"])

    ## defining model checkpointing callbacks

    artifacts = config["artifacts"]
    CKPT_dir = os.path.join(artifacts["artifacts_dir"], artifacts["checkpoint_dir"])
    os.makedirs(CKPT_dir, exist_ok=True )

    CKPT_path = os.path.join(CKPT_dir,'model_ckpt.h5')

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]

    return CALLBACKS_LIST
