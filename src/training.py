from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_model,save_plot

import argparse
import os


def training(config_path):
    config = read_config(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]
    EPOCHS = config["params"]["epochs"]
    BATCH_SIZE = config["params"]["batch_size"]
    
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = get_data(validation_datasize)

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    VALIDATION_SET = (x_valid, y_valid)

    model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_data= VALIDATION_SET, batch_size=BATCH_SIZE)

    model_name = config["artifacts"]["model_name"]
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)

    plot_name = config["artifacts"]["plot_name"]
    plots_dir = config["artifacts"]["plots_dir"]

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plots_dir_path, exist_ok=True)

    save_plot(model_history, plot_name, plots_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)
