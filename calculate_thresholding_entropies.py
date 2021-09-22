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

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    intensities = [i for i in range(256)]

    string_dsc = [','.join(['image'] + [str(i) for i in intensities])]
    string_hr = [','.join(['image'] + [str(i) for i in intensities])]
    n_images = paths.shape[-1]
    scores_dsc = np.zeros((n_images, len(intensities)))
    scores_hr = np.zeros((n_images, len(intensities)))

    intensities = np.array(intensities, dtype=np.float)
    for i, path in enumerate(paths.transpose()):
        img_path = path[0]
        gt_path = path[1]

        dir_path, name = os.path.split(img_path)
        dir_name = os.path.split(dir_path)[-1]
        current_string_dsc = [os.path.join(dir_name, name)] + ['' for n in range(len(intensities))]
        current_string_hr = [os.path.join(dir_name, name)] + ['' for n in range(len(intensities))]

        x_color = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.dtype == np.uint8:
            gt = gt.astype(np.float) / 255.0

        x = cv2.cvtColor(x_color, cv2.COLOR_BGR2GRAY)
        y = gt

        # (Debugiing purposes)
        h2 = calculate_approximate_Hr2(x, y)
        #

        for idx, intensity in enumerate(intensities):
            if x.dtype.name != "uint8":
                intensity /= 255.0
            y_pred = np.where(x >= intensity, 0.0, 1.0) # Values below intensity are assigned 1.0 since cracks are dark

            dsc = calculate_dsc(y, y_pred)
            scores_dsc[i, idx] = dsc
            current_string_dsc[idx + 1] = "{:.4f}".format(dsc)

            hr = calculate_Hr(x, y_pred)
            scores_hr[i, idx] = hr
            current_string_hr[idx + 1] = "{:.4f}".format(hr)

        string_dsc.append(",".join(current_string_dsc))
        string_hr.append(",".join(current_string_hr))

    # Save the thresholding DSC
    current_string_dsc = []
    summary_string_dsc = []
    average_dsc = np.average(scores_dsc, axis=0)
    for i, score in enumerate(average_dsc):
        current_string_dsc.append("{:.4f}".format(score))
        summary_string_dsc.append("{}: {:.4f}".format(intensities[i], score))
    current_string_dsc = ['average'] + current_string_dsc
    string_dsc.append(",".join(current_string_dsc))

    summary_dsc = "\n".join(summary_string_dsc)
    with open(os.path.join(args.save_to, "dsc_summary.txt"), 'w+') as f:
        f.write(summary_dsc)

    results_dsc = "\n".join(string_dsc)
    with open(os.path.join(args.save_to, "dsc.csv"), 'w+') as f:
        f.write(results_dsc)

    # Save the thresholding Hr
    current_string_hr = []
    summary_string_hr = []
    average_hr = np.average(scores_hr, axis=0)
    for i, score in enumerate(average_hr):
        current_string_hr.append("{:.4f}".format(score))
        summary_string_hr.append("{}: {:.4f}".format(intensities[i], score))
    current_string_hr = ['average'] + current_string_hr
    string_hr.append(",".join(current_string_hr))

    summary_hr = "\n".join(summary_string_hr)
    with open(os.path.join(args.save_to, "hr_summary.txt"), 'w+') as f:
        f.write(summary_hr)

    results_hr = "\n".join(string_hr)
    with open(os.path.join(args.save_to, "hr.csv"), 'w+') as f:
        f.write(results_hr)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_names", type=str, nargs='+',
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'syncrack', 'text'")
    parser.add_argument("-p", "--dataset_paths", type=str, nargs='+',
                        help="Path to the folder containing the dataset as downloaded from the original source.")
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