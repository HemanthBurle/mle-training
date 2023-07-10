import argparse
import logging
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument(
        "-i",
        "--input",
        help="Please give the appropriate input path for the CSV file",
        # choices=["datasets"],
        default="datasets",
    )
    parser.add_argument(
        "-p",
        "--pickle",
        help="Please give the appropriate output path for the pickle file",
        # choices=["artifacts"],
        default="artifacts",
    )
    parser.add_argument(
        "--loglevel",
        help="Please specify the log level",
        choices=["INFO", "DEBUG", "ERROR", "CRITICAL", "WARNING"],
        default="INFO",
    )
    parser.add_argument(
        "--no-console-log",
        help="Please specify the directory for the log file ",
        default="logs",
    )
    parser.add_argument(
        "--logpath",
        help="Please specify the directory for the log file ",
        # choices=["", "logs"],
        default="logs",
    )
    args = parser.parse_args()
    os.makedirs("../../" + args.logpath, exist_ok=True)
    log_dict = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
    }
    if args.logpath == "":
        file_name = ""
    else:
        file_name = "../../" + args.logpath + "/score.log"
    logging.basicConfig(
        level=log_dict[args.loglevel],
        filename=file_name,
        format="%(asctime)s-%(levelname)s-[%(filename)s:%(lineno)s]  %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    ARTIFACTS_PATH = os.path.join("../../", args.pickle)
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "lin_reg_op.pkl")
    with open(PICKLE_PATH, "rb") as file:
        lin_reg = pickle.load(file)

    PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "tree_reg_op.pkl")
    with open(PICKLE_PATH, "rb") as file:
        tree_reg = pickle.load(file)

    PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "cvres.pkl")
    with open(PICKLE_PATH, "rb") as file:
        cvres = pickle.load(file)

    PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "new_cvres.pkl")
    with open(PICKLE_PATH, "rb") as file:
        new_cvres = pickle.load(file)

def model_score_calculation():
    """This function calculates the score of the models

    Parameters
    ----------

    Returns
    -------
                prints the score of all models

    """
    # linear regression scores
    print("\nLinear regression MAE:", lin_reg[1], " RMSE:", lin_reg[0])
    # decision tree rmse
    print("\nDecision tree RMSE:", tree_reg)

    # random forest rmse
    print("\nRandom forest regression RMSE:")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    # random forest rmse
    print("\nModified Random forest regression RMSE:")
    for mean_score, params in zip(
        new_cvres["mean_test_score"], new_cvres["params"]
    ):
        print(np.sqrt(-mean_score), params)

    logging.info("The score calculation was successful")

model_score_calculation()
