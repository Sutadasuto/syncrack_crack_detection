import argparse
import cv2
import numpy as np
import os

import data

from distutils.util import strtobool
from scores import *


def main(args):
    # Here we find the paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    predicted_paths = data.create_image_paths(args.dataset_names, args.prediction_paths)

    n_datesets = len(args.dataset_paths)

    score_names = ['dice_score_coefficient', 'precision', 'recall',
                   'crack_region_entropy', 'background_region_entropy', 'entropy',
                   'crack_region_second_order_entropy', 'background_region_second_order_entropy',
                   'second_order_entropy',
                   'kolmogorov-smirnov_alpha5']
    string_list = [','.join(['image'] + score_names)]
    n_images = paths.shape[-1]
    scores = np.zeros((n_images, len(score_names)))

    bar_size = 10
    for i, path in enumerate(paths.transpose()):
        n_stars = round(bar_size * (i + 1) / n_images)
        print("\r[" + "*" * n_stars + " " * (bar_size - n_stars) + "]", end='')
        img_path = path[0]
        gt_path = path[1]
        score_dict = {}

        pred_path = predicted_paths[1, i]
        dir_path, name = os.path.split(pred_path)
        dir_name = os.path.split(dir_path)[-1]
        current_string_list = [os.path.join(dir_name, name)] + ['' for n in score_names]
        if not os.path.split(gt_path)[-1] == os.path.split(pred_path)[-1]:
            raise ValueError("The predicted file name doesn't match the GT file name")

        x_color = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.dtype == np.uint8:
            gt = gt.astype(np.float) / 255.0
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred.dtype == np.uint8:
            pred = pred.astype(np.float) / 255.0

        x = cv2.cvtColor(x_color, cv2.COLOR_BGR2GRAY)
        y = gt
        y_pred = np.where(pred >= 0.5, 1.0, 0.0)

        score_dict['dice_score_coefficient'] = calculate_dsc(y, y_pred)

        confusion_matrix, matrix_names = calculate_confusion_matrix(y, y_pred)
        score_dict['precision'], score_dict['recall'], f = calculate_PRF(confusion_matrix)

        score_dict['crack_region_entropy'] = calculate_Hr(x, y_pred)
        score_dict['background_region_entropy'] = calculate_Hr(x, 1 - y_pred)
        score_dict['entropy'] = score_dict['crack_region_entropy'] + score_dict['background_region_entropy']

        score_dict['crack_region_second_order_entropy'] = calculate_approximate_Hr2(x, y_pred)
        score_dict['background_region_second_order_entropy'] = calculate_approximate_Hr2(x, 1 - y_pred)
        score_dict['second_order_entropy'] = score_dict['crack_region_second_order_entropy'] + score_dict[
            'background_region_second_order_entropy']

        score_dict['kolmogorov-smirnov_alpha5'] = calculate_kolmogorov_smirnov_statistic(x, y_pred, 5 / 100)

        for key in score_dict.keys():
            scores[i, score_names.index(key)] = score_dict[key]
            current_string_list[score_names.index(key) + 1] = "{:.4f}".format(score_dict[key])
        string_list.append(",".join(current_string_list))
    print("")

    current_string_list = []
    summary_string_list = []
    average_scores = np.average(scores, axis=0)
    for i, score in enumerate(average_scores):
        current_string_list.append("{:.4f}".format(score))
        summary_string_list.append("{}: {:.4f}".format(score_names[i], score))
    current_string_list = ['average'] + current_string_list
    string_list.append(",".join(current_string_list))

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)
    summary_string = "\n".join(summary_string_list)
    with open(os.path.join(args.save_to, "scores_summary.txt"), 'w+') as f:
        f.write(summary_string)

    results_string = "\n".join(string_list)
    with open(os.path.join(args.save_to, "scores.csv"), 'w+') as f:
        f.write(results_string)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_names", type=str, nargs='+',
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'syncrack', 'text'")
    parser.add_argument("-p", "--dataset_paths", type=str, nargs='+',
                        help="Path to the folder containing the dataset as downloaded from the original source.")
    parser.add_argument("-pred", "--prediction_paths", type=str, nargs='+',
                        help="Path to the dataset containing the predictions from the dataset.")
    parser.add_argument("--save_to", type=str, default="results",
                        help="Save results in this location (folder is created if it doesn't exist).")

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None":
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
