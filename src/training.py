from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_model,save_plot
from src.utils.callbacks import getCallbacks

import argparse
import os
import logging
import pandas as pd

logging_str = " [%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir='logs_dir'
general_logs='general_logs'
logs_path = os.path.join(logs_dir,general_logs)

logging.basicConfig(filename=os.path.join(logs_path,'Training-logs.log'), level=logging.INFO, format=logging_str, filemode='w')



def training(config_path):
    ## reading the configuration file
    config = read_config(config_path)

    ## getting value of variable for model training

    validation_datasize = config["params"]["validation_datasize"]
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]
    EPOCHS = config["params"]["epochs"]
    BATCH_SIZE = config["params"]["batch_size"]

    ## preparing data
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = get_data(validation_datasize)

    ## creating model architecture
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    VALIDATION_SET = (x_valid, y_valid)

    ### defining logging dir for tensorboard

    logs_dir = config["logs"]["logs_dir"]
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)


    ## training model
    
    logging.info("Getting the callbacks details...")
    CALLBACKS_LIST = getCallbacks(config,x_train)

    logging.info("Training the model on train data set")
    model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_data= VALIDATION_SET, batch_size=BATCH_SIZE, callbacks=CALLBACKS_LIST)
   
    logging.info("Model Trained successfully")

    ## saving model file
    model_name = config["artifacts"]["model_name"]
    
    save_model(model, model_name, model_dir_path)

    ## plotting model performance
    plot_name = config["artifacts"]["plot_name"]
    plots_dir = config["artifacts"]["plots_dir"]

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plots_dir_path, exist_ok=True)

    save_plot(model_history, plot_name, plots_dir_path)


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    logging.info(">>>>>>>>>>>>>>>>>>>>>>>>> Starting of training Script >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    training(config_path = parsed_args.config)

    logging.info(" <<<<<<<<<<<<<<<<<<<< Training Completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
