from __future__ import absolute_import
import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime

from pgportfolio.tools.configprocess import preprocess_config
from pgportfolio.tools.configprocess import load_config
from pgportfolio.tools.trade import save_test_data
from pgportfolio.tools.shortcut import execute_backtest
from pgportfolio.resultprocess import plot


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, generate, download_data"
                             " backtest",
                        metavar="MODE", default="download_data")
    parser.add_argument("--processes", dest="processes",
                        help="number of processes you want to start to train the network",
                        default="1")
    parser.add_argument("--repeat", dest="repeat",
                        help="repeat times of generating training subfolder",
                        default="1")
    parser.add_argument("--algo",
                        help="algo name or indexes of training_package ",
                        dest="algo")
    parser.add_argument("--algos",
                        help="algo names or indexes of training_package, seperated by \",\"",
                        dest="algos")
    parser.add_argument("--labels", dest="labels",
                        help="names that will shown in the figure caption or table header")
    parser.add_argument("--format", dest="format", default="raw",
                        help="format of the table printed")
    parser.add_argument("--device", dest="device", default="gpu",
                        help="device to be used to train")
    parser.add_argument("--folder", dest="folder", type=int,
                        help="folder(int) to load the config, neglect this option if loading from ./pgportfolio/net_config")
    return parser


def _set_logging_by_algo(console_level, file_level, algo, name):
    if algo.isdigit():
            logging.basicConfig(filename="./train_package/"+algo+"/"+name,
                                level=file_level)
            console = logging.StreamHandler()
            console.setLevel(console_level)
            logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(level=console_level)


def _config_by_algo(algo):
    """
    :param algo: a string represent index or algo name
    :return : a config dictionary
    """
    if not algo:
        raise ValueError("please input a specific algo")
    elif algo.isdigit():
        config = load_config(algo)
    else:
        config = load_config()
    return config

# def main():
# parser = build_parser()
# options = parser.parse_args()
    
execute_mode = 'generate'
options_repeat = 1
options_algo = ''
options_folder = '' # Not using
options_processes = 1
options_device = 'gpu'
options_folder = ''


if not os.path.exists("./" + "train_package"):
    os.makedirs("./" + "train_package")
if not os.path.exists("./" + "database"):
    os.makedirs("./" + "database")

if execute_mode == "train":
    import pgportfolio.autotrain.training
    if not options_algo:
        pgportfolio.autotrain.training.train_all(int(options_processes), options_device)
    else:
        for folder in options_folder:
            raise NotImplementedError()
elif execute_mode == "generate":
    import pgportfolio.autotrain.generate as generate
    logging.basicConfig(level=logging.INFO)
    generate.add_packages(load_config(), int(options_repeat))
elif execute_mode == "download_data":
    from pgportfolio.marketdata.datamatrices import DataMatrices
    with open("./pgportfolio/net_config.json") as file:
        config = json.load(file)
    config = preprocess_config(config)
    start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
    end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
    DataMatrices(start=start,
                  end=end,
                  feature_number=config["input"]["feature_number"],
                  window_size=config["input"]["window_size"],
                  online=True,
                  period=config["input"]["global_period"],
                  volume_average_days=config["input"]["volume_average_days"],
                  coin_filter=config["input"]["coin_number"],
                  is_permed=config["input"]["is_permed"],
                  test_portion=config["input"]["test_portion"],
                  portion_reversed=config["input"]["portion_reversed"])
# elif execute_mode == "backtest":
#     config = _config_by_algo(options.algo)
#     _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "backtestlog")
#     execute_backtest(options.algo, config)
# elif execute_mode == "save_test_data":
#     # This is used to export the test data
#     save_test_data(load_config(options.folder))
# elif execute_mode == "plot":
#     logging.basicConfig(level=logging.INFO)
#     algos = options.algos.split(",")
#     if options.labels:
#         labels = options.labels.replace("_"," ")
#         labels = labels.split(",")
#     else:
#         labels = algos
#     plot.plot_backtest(load_config(), algos, labels)
# elif execute_mode == "table":
#     algos = options.algos.split(",")
#     if options.labels:
#         labels = options.labels.replace("_"," ")
#         labels = labels.split(",")
#     else:
#         labels = algos
#     plot.table_backtest(load_config(), algos, labels, format=options.format)
