import cv2
import importlib
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import scipy.io
import subprocess

from math import ceil
from tensorflow.keras.preprocessing import image

from scores import *


### Getting image paths
def create_image_paths(dataset_names, dataset_paths):
    paths = np.array([[], []], dtype=np.str)
    for idx, dataset in enumerate(dataset_names):
        dataset_name = dataset
        dataset_path = dataset_paths[idx]
        if dataset_name == "cfd" or dataset_name == "cfd-pruned" or dataset_name == "cfd-corrected":
            or_im_paths, gt_paths = paths_generator_cfd(dataset_path)
        elif dataset_name == "aigle-rn":
            or_im_paths, gt_paths = paths_generator_crack_dataset(dataset_path, "AIGLE_RN")
        elif dataset_name == "esar":
            or_im_paths, gt_paths = paths_generator_crack_dataset(dataset_path, "ESAR")
        elif dataset_name == "crack500" or dataset_name == "gaps384" or dataset_name == "cracktree200":
            or_im_paths, gt_paths = paths_generator_fphb(dataset_path, dataset_name)
        elif dataset_name == "syncrack":
            or_im_paths, gt_paths = paths_generator_syncrack(dataset_path)
        elif dataset_name == "concrete":
            or_im_paths, gt_paths = paths_generator_concrete(dataset_path)
        elif dataset_name == "text":
            or_im_paths, gt_paths = paths_generator_from_text(dataset_path)

        paths = np.concatenate([paths, [or_im_paths, gt_paths]], axis=-1)
    return paths


def paths_generator_crack_dataset(dataset_path, subset):
    ground_truth_path = os.path.join(dataset_path, "TITS", "GROUND_TRUTH", subset)
    training_data_path = os.path.join(dataset_path, "TITS", "IMAGES", subset)
    images_path, dataset = os.path.split(training_data_path)
    if dataset == "ESAR":
        file_end = ".jpg"
    elif dataset == "AIGLE_RN":
        file_end = "or.png"

    ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and (f.endswith(".png") or f.endswith(".jpg"))],
                                      key=lambda f: f.lower())

    training_image_paths = [os.path.join(training_data_path, "Im_" + os.path.split(f)[-1].replace(".png", file_end)) for
                            f in ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


def paths_generator_cfd(dataset_path):
    ground_truth_path = os.path.join(dataset_path, "groundTruthPng")

    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)

        ground_truth_image_paths = sorted(
            [os.path.join(dataset_path, "groundTruth", f) for f in os.listdir(os.path.join(dataset_path, "groundTruth"))
             if not f.startswith(".") and f.endswith(".mat")],
            key=lambda f: f.lower())
        for idx, path in enumerate(ground_truth_image_paths):
            mat = scipy.io.loadmat(path)
            img = (mat["groundTruth"][0][0][0] - 1).astype(np.float32)
            cv2.imwrite(path.replace("groundTruth", "groundTruthPng").replace(".mat", ".png"), 255 * img)

    ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and f.endswith(".png")],
                                      key=lambda f: f.lower())

    training_image_paths = [os.path.join(dataset_path, "image", os.path.split(f)[-1].replace(".png", ".jpg")) for f in
                            ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


def paths_generator_fphb(dataset_path, dataset_name):
    if dataset_name == "crack500":
        ground_truth_paths = [os.path.join(dataset_path, "traincrop"), os.path.join(dataset_path, "valcrop"),
                              os.path.join(dataset_path, "testcrop")]
        ground_truth_image_paths = []
        for ground_truth_path in ground_truth_paths:
            ground_truth_image_paths += sorted(
                [os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                 if not f.startswith(".") and f.endswith(".png")],
                key=lambda f: f.lower())

        training_image_paths = [f.replace(".png", ".jpg") for f in ground_truth_image_paths]

    elif dataset_name == "gaps384":
        ground_truth_path = os.path.join(dataset_path, "croppedgt")
        image_path = os.path.join(dataset_path, "croppedimg")

        ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                           if not f.startswith(".") and f.endswith(".png")],
                                          key=lambda f: f.lower())
        training_image_paths = sorted([os.path.join(image_path, f.replace(".png", ".jpg"))
                                       for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and f.endswith(".png")],
                                      key=lambda f: f.lower())

    elif dataset_name == "cracktree200":
        ground_truth_path = os.path.join(dataset_path, "cracktree200_gt")
        image_path = os.path.join(dataset_path, "cracktree200rgb")

        ground_truth_image_paths = sorted([os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)
                                           if not f.startswith(".") and f.endswith(".png")],
                                          key=lambda f: f.lower())
        training_image_paths = sorted([os.path.join(image_path, f.replace(".png", ".jpg"))
                                       for f in os.listdir(ground_truth_path)
                                       if not f.startswith(".") and f.endswith(".png")],
                                      key=lambda f: f.lower())

    return training_image_paths, ground_truth_image_paths


def paths_generator_syncrack(dataset_path):
    ground_truth_image_paths = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                                       if not f.startswith(".") and f.endswith("_gt.png")], key=lambda f: f.lower())

    training_image_paths = [os.path.join(dataset_path, os.path.split(f)[1].replace("_gt.png", ".jpg")) for
                            f in ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


def paths_generator_from_text(text_file_path):
    with open(text_file_path, "r") as file:
        lines = file.readlines()
    paths_array = np.concatenate([np.array([line.strip().split(";")]) for line in lines], axis=0)
    return paths_array[:, 0], paths_array[:, 1]


def paths_generator_concrete(dataset_path):
    ground_truth_image_paths = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                                       if not f.startswith(".") and f.endswith("_gt.png")], key=lambda f: f.lower())
    base_name = os.path.split(ground_truth_image_paths[0])[-1].replace("_gt.png", "")
    for f in os.listdir(dataset_path):
        if f.startswith(base_name) and not f.endswith("_gt.png"):
            extension = os.path.splitext(f)[-1]
            break

    training_image_paths = [os.path.join(dataset_path, os.path.split(f)[1].replace("_gt.png", extension)) for
                            f in ground_truth_image_paths]

    return training_image_paths, ground_truth_image_paths


def paths_generator_from_text(text_file_path):
    with open(text_file_path, "r") as file:
        lines = file.readlines()
    paths_array = np.concatenate([np.array([line.strip().split(";")]) for line in lines], axis=0)
    return paths_array[:, 0], paths_array[:, 1]


### Loading images for Keras

# Utilities
def manual_padding(image, n_pooling_layers):
    # Assuming N pooling layers of size 2x2 with pool size stride (like in U-net and multiscale U-net), we add the
    # necessary number of rows and columns to have an image fully compatible with up sampling layers.
    divisor = 2 ** n_pooling_layers
    try:
        h, w = image.shape
    except ValueError:
        h, w, c = image.shape
    new_h = divisor * ceil(h / divisor)
    new_w = divisor * ceil(w / divisor)
    if new_h == h and new_w == w:
        return image

    if new_h != h:
        new_rows = np.flip(image[h - new_h:, :, ...], axis=0)
        image = np.concatenate([image, new_rows], axis=0)
    if new_w != w:
        new_cols = np.flip(image[:, w - new_w:, ...], axis=1)
        image = np.concatenate([image, new_cols], axis=1)
    return image


def get_corners(im, input_size):
    h, w, c = im.shape
    rows = h / input_size[0]
    cols = w / input_size[0]

    corners = []
    for i in range(ceil(rows)):
        for j in range(ceil(cols)):
            if i + 1 <= rows:
                y = i * input_size[0]
            else:
                y = h - input_size[0]

            if j + 1 <= cols:
                x = j * input_size[1]
            else:
                x = w - input_size[1]

            corners.append([y, x])
    return corners


def crop_generator(im, gt, input_size):
    corners = get_corners(im, input_size)
    for corner in corners:
        x = im[corner[0]:corner[0] + input_size[0], corner[1]:corner[1] + input_size[1], ...]
        y = gt[corner[0]:corner[0] + input_size[0], corner[1]:corner[1] + input_size[1], ...]
        yield [x, y]


# Data augmentation
def random_transformation(im, gt, **kwargs):
    noise = random.choice(kwargs["noises"])
    alpha = random.choice(kwargs["alphas"])
    beta = random.choice(kwargs["betas"])
    flip = random.choice(kwargs["flips"])
    zoom = random.choice(kwargs["zooms"])
    rot_ang = random.choice(kwargs["rot_angs"])
    shear_ang = random.choice(kwargs["shear_angs"])

    noisy = noisy_version(im, noise)
    adjusted = illumination_adjustment_version(noisy, alpha, beta)

    flipped = flipped_version(adjusted, flip)
    flipped_gt = flipped_version(gt, flip)

    affine_transformed = image.apply_affine_transform(flipped, rot_ang, shear=shear_ang, zx=zoom, zy=zoom,
                                                      fill_mode="reflect")
    affine_transformed_gt = image.apply_affine_transform(flipped_gt, rot_ang, shear=shear_ang, zx=zoom, zy=zoom,
                                                         fill_mode="reflect")

    return affine_transformed, np.where(affine_transformed_gt > 0.5, 1.0, 0.0)


def flipped_version(image, flip_typ):
    if flip_typ is None:
        return image
    elif flip_typ == "h":
        return np.fliplr(image)
    elif flip_typ == "v":
        return np.flipud(image)


def noisy_version(image, noise_typ):
    int_type = True if image.dtype == np.uint8 else False
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1 * image.max()
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy if not int_type else noisy.astype(np.uint8)

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        coords = [np.random.randint(0, i - 1, num_salt)
                  for i in image.shape[:-1]]
        for channel in range(image.shape[-1]):
            channel_coords = tuple(coords + [np.array([channel for i in range(num_salt)])])
            out[channel_coords] = image.max()

        # Pepper mode
        num_pepper = int(np.ceil(amount * image.size * (1. - s_vs_p)))
        coords = [np.random.randint(0, i - 1, num_pepper)
                  for i in image.shape[:-1]]
        for channel in range(image.shape[-1]):
            channel_coords = tuple(coords + [np.array([channel for i in range(num_pepper)])])
            out[channel_coords] = image.min()
        return out if not int_type else out.astype(np.uint8)

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch) / 4.0
        noisy = image + image * gauss
        return noisy if not int_type else noisy.astype(np.uint8)

    elif noise_typ is None:
        return image


def illumination_adjustment_version(image, alpha, beta):
    image = alpha * image
    if beta == "bright":
        shift = 255 - np.max(image)
    elif beta == "dark":
        shift = -np.min(image)
    else:
        shift = 0
    return np.clip(image + shift, 0, 255).astype(np.uint8)


def rotated_version(image, angle):
    if angle is None:
        return image

    k = int(angle / 90)
    return np.rot90(image, k)


# Image generators
def get_image(im_path):
    # im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    ### Color debugging
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)[..., None]
    im = np.concatenate([im, im, im], axis=-1)
    ###

    im = manual_padding(im, n_pooling_layers=4)
    if len(im.shape) == 2:
        im = im[..., None]  # Channels last
    return im


def get_gt_image(gt_path):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    binary = True if len(np.unique(gt)) <= 2 else False
    gt = (gt / 255.0)
    if binary:
        n_white = np.sum(gt)
        n_black = gt.shape[0] * gt.shape[1] - n_white
        if n_black < n_white:
            gt = 1 - gt

    gt = manual_padding(gt, n_pooling_layers=4)
    if len(gt.shape) < 3:
        return gt[..., None]  # Channels last
    else:
        return gt


def validation_image_generator(paths, batch_size=1, rgb_preprocessor=None):
    _, n_images = paths.shape
    rgb = True if rgb_preprocessor else False
    i = 0
    while True:
        batch_x = []
        batch_y = []
        b = 0
        while b < batch_size:
            if i == n_images:
                i = 0
            im_path = paths[0][i]
            gt_path = paths[1][i]

            im = get_image(im_path)
            gt = get_gt_image(gt_path)
            if rgb:
                batch_x.append(rgb_preprocessor(im))
            else:
                batch_x.append(im)
            batch_y.append(gt)
            b += 1
            i += 1

        yield np.array(batch_x), np.array(batch_y)


# This version applies a random transformation to each input pair before feeding it to the model
def train_image_generator(paths, input_size, batch_size=1, count_samples_mode=False,
                          rgb_preprocessor=None, data_augmentation=True):
    _, n_images = paths.shape
    rgb = True if rgb_preprocessor else False

    # All available transformations for data augmentation
    if data_augmentation:
        augmentation_params = {
            "noises": [None, "gauss", "s&p"],
            "alphas": [1.0, 0.8],  # Simple contrast control
            "betas": [None, "bright", "dark"],  # Simple brightness control
            "flips": [None, "h", "v"],  # Horizontal, Vertical
            "zooms": [1.0, 2.0, 1.5],  # Zoom rate in both axis; >1.0 zooms out
            "rot_angs": [i * 5.0 for i in range(int(90.0 / 5.0 + 1))],  # Rotation angle in degrees
            "shear_angs": [i * 5.0 for i in range(int(45.0 / 5.0 + 1))]  # Shear angle in degrees
        }

    # This means no noise, no illumination adjustment, no rotation and no flip (i.e. only the original image is
    # provided)
    else:
        augmentation_params = {
            "noises": [None],  # s&p = salt and pepper
            "alphas": [1.0],  # Simple contrast control
            "betas": [None],  # Simple brightness control
            "flips": [None],  # Horizontal, Vertical
            "zooms": [1.0],  # Zoom rate in both axis; >1.0 zooms out
            "rot_angs": [0.0],  # Degrees
            "shear_angs": [0.0],  # Degrees
        }

    i = -1
    prev_im = False
    n_samples = 0

    while True:
        batch_x = []
        batch_y = []
        b = 0
        while b < batch_size:

            if not prev_im:
                i += 1

                if i == n_images:
                    if count_samples_mode:
                        yield n_samples
                    i = 0
                    np.random.shuffle(paths.transpose())

                if count_samples_mode:
                    print("\r%s/%s paths analyzed so far" % (str(i + 1).zfill(len(str(n_images))), n_images), end='')

                im_path = paths[0][i]
                gt_path = paths[1][i]

                or_im = get_image(im_path)
                or_gt = get_gt_image(gt_path)

            if input_size:
                if not prev_im:
                    win_gen = crop_generator(or_im, or_gt, input_size)
                    prev_im = True
                try:
                    [im, gt] = next(win_gen)
                except StopIteration:
                    prev_im = False
                    continue

                x, y = random_transformation(im, gt, **augmentation_params)

                x = manual_padding(x, 4)
                y = manual_padding(y, 4)
                if rgb:
                    batch_x.append(rgb_preprocessor(x))
                else:
                    batch_x.append(x)
                batch_y.append(y)
                n_samples += 1
                b += 1

            else:
                im, gt = random_transformation(or_im, or_gt, **augmentation_params)
                im = manual_padding(im, 4)
                if rgb:
                    batch_x.append(rgb_preprocessor(im))
                else:
                    batch_x.append(im)
                gt = manual_padding(gt, 4)
                batch_y.append(gt)
                n_samples += 1
                b += 1

        if not count_samples_mode:
            yield np.array(batch_x), np.array(batch_y)


# To test the model on images
def get_preprocessor(model):
    """
    :param model: A Tensorflow model
    :return: A preprocessor corresponding to the model name
    Model name should match with the name of a model from
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/
    This assumes you used a model with RGB inputs as the first part of your model,
    therefore your input data should be preprocessed with the corresponding
    'preprocess_input' function.
    If the model model is not part of the keras applications models, None is returned
    """
    try:
        m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
        return getattr(m, "preprocess_input")
    except ModuleNotFoundError:
        return None


### Make results analysis

# Compare GT and predictions from images obtained by save_results_on_paths()
def highlight_cracks(or_im, mask, bg_color, fade_intensity):
    highlight_mask = np.zeros(mask.shape, dtype=np.float)
    if bg_color == "black":
        highlight_mask[np.where(mask >= 128)] = 1.0
        highlight_mask[np.where(mask < 128)] = fade_intensity
    else:
        highlight_mask[np.where(mask >= 128)] = fade_intensity
        highlight_mask[np.where(mask < 128)] = 1.0
    return or_im * highlight_mask


def compare_masks(gt_mask, pred_mask, bg_color):
    if bg_color == "black":
        new_image = np.zeros(gt_mask.shape, dtype=np.float32)
        new_image[..., 2][np.where(pred_mask[..., 0] >= 128)] = 255
        new_image[..., 0][np.where(gt_mask[..., 0] >= 128)] = 255
        new_image[..., 1][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 255
        new_image[..., 0][np.where((new_image[..., 0] == 255) & (new_image[..., 2] == 255))] = 0
        new_image[..., 2][
            np.where((new_image[..., 0] == 0) & (new_image[..., 1] == 255) & (new_image[..., 2] == 255))] = 0
    else:
        new_image = 255 * np.ones(gt_mask.shape, dtype=np.float32)
        new_image[..., 0][np.where(pred_mask[..., 0] < 128)] = 0
        new_image[..., 2][np.where(gt_mask[..., 0] < 128)] = 0
        new_image[..., 1][
            np.where((new_image[..., 0]) == 0 & (new_image[..., 1] == 255) & (new_image[..., 2] == 255))] = 0
        new_image[..., 1][
            np.where((new_image[..., 2]) == 0 & (new_image[..., 0] == 255) & (new_image[..., 1] == 255))] = 0
        new_image[..., 1][
            np.where((new_image[..., 0] == 0) & (new_image[..., 1] == 0) & (new_image[..., 2] == 0))] = 255

    return new_image


def test_image_from_path(model, input_path, gt_path, rgb_preprocessor=None, verbose=0):
    if rgb_preprocessor is None:
        rgb_preprocessor = get_preprocessor(model)
    rgb = True if rgb_preprocessor else False
    if rgb:
        prediction = model.predict(
            rgb_preprocessor(get_image(input_path))[None, ...], verbose=verbose)[0, ...]

    if gt_path:
        gt = get_gt_image(gt_path)[..., 0]
    input_image = cv2.cvtColor(get_image(input_path), cv2.COLOR_BGR2GRAY)[..., None] / 255.0
    if gt_path:
        return [input_image, gt, prediction]
    return [input_image, None, prediction]


def evaluate_model_on_paths(model, paths, output_folder, args):
    prediction_folder = os.path.join(output_folder, "predictions")
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    evaluation_folder = os.path.join(output_folder, "evaluation")
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)

    n_images = paths.shape[-1]
    input_images = ['' for i in range(n_images)]
    pred_images = ['' for i in range(n_images)]
    dir_names = []
    for path in paths.transpose():
        gt_path = path[1]
        dir_path, name = os.path.split(gt_path)
        dir_name = os.path.split(dir_path)[-1]
        if not (dir_name in dir_names):
            dir_names.append(dir_name)
    if len(dir_names) > 1:
        for dir_name in dir_names:
            if not os.path.exists(os.path.join(prediction_folder, dir_name)):
                os.makedirs(os.path.join(prediction_folder, dir_name))
            if not os.path.exists(os.path.join(evaluation_folder, dir_name)):
                os.makedirs(os.path.join(evaluation_folder, dir_name))
    print("Saving predictions...")
    bar_size = 10
    for i, path in enumerate(paths.transpose()):
        n_stars = round(bar_size * (i + 1) / n_images)
        print("\r[" + "*" * n_stars + " " * (bar_size - n_stars) + "]", end='')

        img_path = path[0]
        gt_path = path[1]
        input_images[i] = "%s;%s" % (img_path, gt_path)

        dir_path, name = os.path.split(gt_path)
        dir_name = os.path.split(dir_path)[-1]
        name, extension = os.path.splitext(name)
        # extension = ".png"

        [im, gt, pred] = test_image_from_path(model, img_path, gt_path, rgb_preprocessor=None)

        x_color = cv2.imread(img_path)
        or_shape = x_color.shape
        gt = gt[:or_shape[0], :or_shape[1]]
        pred = pred[:or_shape[0], :or_shape[1], 0]

        # x = cv2.cvtColor(x_color, cv2.COLOR_BGR2GRAY)
        y = gt
        y_pred = np.where(pred >= 0.5, 1.0, 0.0)

        if len(dir_names) > 1:
            pred_path = os.path.join(prediction_folder, dir_name, "%s%s" % (name, extension))
        else:
            pred_path = os.path.join(prediction_folder, "%s%s" % (name, extension))
        cv2.imwrite(pred_path, 255 * pred)
        pred_images[i] = "%s;%s" % (img_path, pred_path)

        y_color = 255 * np.concatenate([y[..., None] for c in range(3)], axis=-1).astype(np.uint8)
        y_pred_color = 255 * np.concatenate([y_pred[..., None] for c in range(3)], axis=-1).astype(np.uint8)
        pred_comparison = compare_masks(255 - y_color, 255 - y_pred_color, bg_color='white').astype(np.uint8)
        comparative_image = np.concatenate([x_color, y_color, y_pred_color, pred_comparison], axis=1)
        if len(dir_names) > 1:
            comparison_path = os.path.join(evaluation_folder, dir_name, "%s%s" % (name, extension))
        else:
            comparison_path = os.path.join(evaluation_folder, "%s%s" % (name, extension))
        cv2.imwrite(comparison_path, comparative_image)
    print("")

    input_text_dataset_path = "temp_input.txt"
    with open(input_text_dataset_path, 'w+') as f:
        f.write("\n".join(input_images))
    pred_text_dataset_path = "temp_output.txt"
    with open(pred_text_dataset_path, 'w+') as f:
        f.write("\n".join(pred_images))
    command = "python calculate_scores.py -d %s -p %s -pred %s --save_to %s" % \
              ("text", input_text_dataset_path, pred_text_dataset_path, evaluation_folder)
    print("Calculating and saving scores...")
    subprocess.run(command, shell=True)
    os.remove(input_text_dataset_path)
    os.remove(pred_text_dataset_path)

    parameters_string = "\n"
    for attribute in args.__dict__.keys():
        parameters_string += "\n--%s: %s" % (attribute, str(args.__getattribute__(attribute)))
    with open(os.path.join(evaluation_folder, "scores_summary.txt"), 'a') as f:
        f.write(parameters_string)


### Interaction with Syncrack generator
def generate_datasets(path_to_syncrack_generator, args_dict, destination_folder, category_name, noise_levels):
    generator_script = os.path.join(path_to_syncrack_generator, "generate_dataset.py")
    noise_generator_script = os.path.join(path_to_syncrack_generator, "generate_noisy_labels.py")
    dataset_root_dir = os.path.join(destination_folder, category_name)

    for i, level in enumerate(noise_levels):
        if level == 0:
            args_dict["-f"] = os.path.join(dataset_root_dir, "%s" % category_name)
            args = " ".join(["%s %s" % (key, args_dict[key]) for key in args_dict.keys()])
            command = "python %s %s" % (generator_script, args)
            subprocess.run(command, shell=True)
            command = "python calculate_scores.py -d syncrack -p %s -pred %s --save_to %s" % \
                      (args_dict["-f"], args_dict["-f"], args_dict["-f"])
            subprocess.run(command, shell=True)
        else:
            new_folder_name = os.path.join(dataset_root_dir, "%s_%s_attacked" % (category_name, level))
            command = "python %s %s -np %s --save_to %s" % \
                      (noise_generator_script, args_dict["-f"], level / 100, new_folder_name)
            subprocess.run(command, shell=True)
            command = "python calculate_scores.py -d syncrack -p %s -pred %s --save_to %s" % \
                      (args_dict["-f"], new_folder_name, new_folder_name + "_label_comparison")
            subprocess.run(command, shell=True)


def generate_dataset_summary(datasets_folder, category, noise_levels):
    noise_level_dicts = []
    for noise_level in noise_levels:
        # Evaluation scores (e.g. Dice Score Coefficient, Entropy, etc.)
        if noise_level == 0:
            quality_scores_file_path = os.path.join(datasets_folder, category, category, "scores.csv")
        else:
            quality_scores_file_path = os.path.join(datasets_folder, category, "%s_%s_attacked_label_comparison" %
                                                    (category, noise_level), "scores.csv")
        with open(quality_scores_file_path, "r") as f:
            quality_scores = f.readlines()
        if noise_level == 0:
            quality_score_names = quality_scores[0].strip().split(",")[1:]
        image_dict = {os.path.split(line.split(",")[0])[-1]:  # Get image name
                          {quality_score_names[
                               idx]:  # Assign to each image its scores for 0 noise as dict
                           # idx moves along the scores
                               line.strip().split(",")[1:][idx] for idx in
                           range(len(quality_score_names))
                           }
                      for line in quality_scores[1:-1]  # line is each line from the scores file
                      }

        # Confusion matrix (True positives, false positives, true negatives, false negatives)
        if noise_level == 0:
            comparison_scores_file_path = os.path.join(datasets_folder, category, category, "%s-VS-%s.csv" %
                                                       (category, category))
        else:
            comparison_scores_file_path = os.path.join(datasets_folder, category,
                                                       "%s_%s_attacked_label_comparison" % (category, noise_level),
                                                       "%s-VS-%s_%s_attacked.csv" % (
                                                           category, category, noise_level))
        with open(comparison_scores_file_path, "r") as f:
            comparison_scores = f.readlines()
        if noise_level == 0:
            comparison_score_names = comparison_scores[0].split(",")[1:-1]
        for line in comparison_scores[1:-2]:
            for idx in range(len(comparison_score_names)):
                image_dict[os.path.split(line.split(",")[0])[-1]][comparison_score_names[idx]] = \
                    line.strip().split(",")[1:][idx]

        noise_level_dicts.append(image_dict)

    header = [["", "", "noise_percentage"] + ["" for i in range(len(noise_levels) - 1)],
              ["image", "metric"] + [str(level) for level in noise_levels]]
    image_names = noise_level_dicts[0].keys()  # Number of images at 0 percentage noise
    # Number of values for the first image at 0 percentage noise
    scores = noise_level_dicts[0][list(noise_level_dicts[0].keys())[0]].keys()
    image_list = []
    for name in image_names:
        for s_idx, score in enumerate(scores):
            first_columns = [os.path.join(category, name), score] if s_idx == 0 else ["", score]
            image_list.append(first_columns +
                              [noise_level_dicts[l_idx][name][score] for l_idx, level in enumerate(noise_levels)])
    with open(os.path.join(datasets_folder, category, "all_scores.csv"), "w+") as f:
        f.write("\n".join([",".join(line) for line in (header + image_list)]))

    # Create summary of noise percentages only (confusion matrices)
    confusion_matrices = []
    for noise_level in noise_levels:
        # Confusion matrix (True positives, false positives, true negatives, false negatives)
        if noise_level == 0:
            comparison_scores_file_path = os.path.join(datasets_folder, category, category, "%s-VS-%s.csv" %
                                                       (category, category))
        else:
            comparison_scores_file_path = os.path.join(datasets_folder, category,
                                                       "%s_%s_attacked_label_comparison" % (category, noise_level),
                                                       "%s-VS-%s_%s_attacked.csv" % (
                                                           category, category, noise_level))
        with open(comparison_scores_file_path, "r") as f:
            comparison_scores = f.readlines()
        if noise_level == 0:
            comparison_score_names = comparison_scores[0].split(",")[1:-1]
        confusion_matrices.append(
            comparison_scores[-1].split(",")[1:-1])  # Get only the dataset percentages (TP,FP,TN,FN)

    header = np.array([["", "noise_percentage"] + ["" for i in range(len(noise_levels) - 1)],
                       ["metric"] + [str(level) for level in noise_levels]])
    confusion_matrices = np.array(confusion_matrices).transpose()
    comparison_score_names = np.array([comparison_score_names]).transpose()
    confusion_matrices = np.concatenate((comparison_score_names, confusion_matrices), axis=1)
    confusion_matrices = np.concatenate((header, confusion_matrices), axis=0)
    with open(os.path.join(datasets_folder, category, "confusion_matrices.csv"), "w+") as f:
        f.write("\n".join([",".join(line) for line in confusion_matrices.tolist()]))


def generate_plots(supervised_metrics, unsupervised_metrics, noise_levels, datasets_folder, category, zoom=False,
                   order=False):
    noise_level_dicts = []
    for noise_level in noise_levels:
        # Evaluation scores (e.g. Dice Score Coefficient, Entropy, etc.)
        if noise_level == 0:
            quality_scores_file_path = os.path.join(datasets_folder, category, category, "scores.csv")
        else:
            quality_scores_file_path = os.path.join(datasets_folder, category, "%s_%s_attacked_label_comparison" %
                                                    (category, noise_level), "scores.csv")
        with open(quality_scores_file_path, "r") as f:
            quality_scores = f.readlines()
        if noise_level == 0:
            quality_score_names = quality_scores[0].strip().split(",")[1:]
        image_dict = {os.path.split(line.split(",")[0])[-1]:  # Get image name
                          {quality_score_names[
                               idx]:  # Assign to each image its scores for 0 noise as dict
                           # idx moves along the scores
                               line.strip().split(",")[1:][idx] for idx in
                           range(len(quality_score_names))
                           }
                      for line in quality_scores[1:-1]  # line is each line from the scores file
                      }
        noise_level_dicts.append(image_dict)
    image_names = noise_level_dicts[0].keys()

    supervised_scores = np.zeros((len(image_names), len(supervised_metrics), len(noise_levels)))
    unsupervised_scores = np.zeros((len(image_names), len(unsupervised_metrics), len(noise_levels)))
    for n, noise_dict in enumerate(noise_level_dicts):
        for i, image in enumerate(image_names):
            for m, metric in enumerate(supervised_metrics):
                supervised_scores[i, m, n] = float(noise_dict[image][metric])
    for n, noise_dict in enumerate(noise_level_dicts):
        for i, image in enumerate(image_names):
            for m, metric in enumerate(unsupervised_metrics):
                unsupervised_scores[i, m, n] = float(noise_dict[image][metric])
    supervised_averages = np.average(supervised_scores, axis=0)
    supervised_std = np.std(supervised_scores, axis=0)
    unsupervised_averages = np.average(unsupervised_scores, axis=0)
    unsupervised_std = np.std(unsupervised_scores, axis=0)

    fig, ax = plt.subplots()
    ax.set_ylabel('score')
    ax.set_xlabel('noise')
    ax.set_title('Scores by noise level')
    ax.set_xticks(noise_levels)
    ax.set_ylim([0, 1])
    for m, metric in enumerate(supervised_metrics):
        x = np.array(noise_levels)
        y = supervised_averages[m, :]
        e = supervised_std[m, :]

        ax.errorbar(x, y, e, marker='o', label=metric)
        ax.legend()
    plt.savefig(os.path.join(datasets_folder, category, "supervised_scores_along_noise.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_ylabel('score')
    ax.set_xlabel('noise')
    ax.set_title('Scores by noise level')
    ax.set_xticks(noise_levels)
    ax.set_ylim([0, 9])
    for m, metric in enumerate(unsupervised_metrics):
        x = np.array(noise_levels)
        y = unsupervised_averages[m, :]
        e = unsupervised_std[m, :]

        ax.errorbar(x, y, e, marker='o', label=metric)
        ax.legend()
    plt.savefig(os.path.join(datasets_folder, category, "unsupervised_scores_along_noise.png"))
    plt.close(fig)

    if zoom:
        for m, metric in enumerate(unsupervised_metrics):
            fig, ax = plt.subplots()
            ax.set_ylabel(metric)
            ax.set_xlabel('noise')
            ax.set_xticks(noise_levels)

            x = np.array(noise_levels)
            y = unsupervised_averages[m, :]
            e = unsupervised_std[m, :]

            ax.errorbar(x, y, e, marker='o', label=metric)
            ax.legend()
            plt.savefig(os.path.join(datasets_folder, category, "%s_along_noise.png" % metric))
            plt.close(fig)

    if order:
        for m, metric in enumerate(supervised_metrics):
            for n, noise in enumerate(noise_levels):
                current_noise_scores = supervised_scores[:, :, n]
                order = current_noise_scores[:, m].argsort()
                ordered_scores = current_noise_scores[order, :]

                fig, ax = plt.subplots()
                ax.set_ylabel('score')
                ax.set_xlabel('image')
                ax.set_title('Scores in images sorted by increasing %s with noise level %s' % (metric, noise))
                ax.set_ylim([0, 8.5])
                for um, u_metric in enumerate(unsupervised_metrics):
                    x = np.array([i + 1 for i in range(len(order))])
                    y = unsupervised_scores[order, um, n]

                    ax.plot(x, y, linewidth=1, marker=',', label=u_metric)
                    ax.legend()
                plt.savefig(os.path.join(datasets_folder, category,
                                         "unsupervised_scores_images_sorted_by_%s_with_noise_%s.png" % (metric, noise)))
                plt.close(fig)

                if zoom:
                    for um, u_metric in enumerate(unsupervised_metrics):
                        fig, ax = plt.subplots()
                        ax.set_ylabel(u_metric)
                        ax.set_xlabel('image')
                        if u_metric == "crack_region_entropy":
                            ax.set_ylim([3, 4.5])
                        elif u_metric == "crack_region_second_order_entropy":
                            ax.set_ylim([5, 8.5])
                        elif u_metric == "kolmogorov-smirnov_alpha5":
                            ax.set_ylim([0, 1])
                        x = np.array([i + 1 for i in range(len(order))])
                        y = unsupervised_scores[order, um, n]

                        ax.plot(x, y, linewidth=1, marker=',', label=u_metric)
                        ax.legend()
                        plt.savefig(os.path.join(datasets_folder, category,
                                                 "%s_images_sorted_by_%s_with_noise_%s.png" % (u_metric, metric, noise)))
                    plt.close(fig)


def train_synthetic_model(datasets_folder, results_root, synthetic_training_parameters, noise_level, category):
    if noise_level == 0:
        p = os.path.join(datasets_folder, category, "%s" % category)
        cp = "None"
    else:
        p = os.path.join(datasets_folder, category, "%s_%s_attacked" % (category, noise_level))
        cp = os.path.join(datasets_folder, category, "%s" % category)
    save_to = os.path.join(results_root, "results_%s" % category, "%s_percent_noise" % noise_level)

    args = " ".join(["%s %s" % (key, synthetic_training_parameters[key])
                     for key in synthetic_training_parameters.keys()])
    command = "python train_and_validate.py -d %s -p %s -cp %s %s --save_to %s" % ("syncrack", p, cp, args, save_to)
    print("Running: '%s'" % command)
    subprocess.run(command, shell=True)
    print("Done running: '%s'" % command)


def create_syncthetic_training_summary(results_root, categories, noise_levels):
    summary_text_path = os.path.join(results_root, "results_%s" % categories[0], "%s_percent_noise" % noise_levels[0],
                                     "results_validation_min_val_loss", "evaluation", "scores_summary.txt")
    with open(summary_text_path, 'r') as f:
        text = f.read()
    scores = text.split('\n\n')[0].split('\n')
    score_names = [score.split(': ')[0] for score in scores]
    n_scores = len(score_names)
    results_grid = np.zeros((len(categories) * n_scores, len(noise_levels))).astype(np.str)

    for i, c in enumerate(categories):
        for j, l in enumerate(noise_levels):

            if noise_levels[j] == 0:
                summary_text_path = os.path.join(results_root, "results_%s" % categories[i],
                                                 "%s_percent_noise" % noise_levels[j],
                                                 "results_validation_min_val_loss", "evaluation", "scores_summary.txt")
            else:
                summary_text_path = os.path.join(results_root, "results_%s" % categories[i],
                                                 "%s_percent_noise" % noise_levels[j],
                                                 "results_clean", "evaluation", "scores_summary.txt")
            with open(summary_text_path, 'r') as f:
                text = f.read()
            scores = text.split('\n\n')[0].split('\n')
            score_values = [score.split(': ')[1] for score in scores]
            for k, score in enumerate(score_values):
                results_grid[i * n_scores + k, j] = score

    categoy_column = [[''] for l in range(len(categories) * n_scores)]
    for l, c in enumerate(categories):
        categoy_column[l * n_scores][0] = c
    metrics_column = np.array([[metric] for metric in score_names] * len(categories))
    body = np.concatenate((categoy_column, metrics_column, results_grid), axis=1)
    noise_header = ["" for l in range(len(noise_levels))]
    noise_header = np.array((noise_header, noise_header)).astype(body.dtype)
    noise_header[0, 0] = 'noise_percentage'
    for l, level in enumerate(noise_levels):
        noise_header[1, l] = level
    header = np.concatenate(([["", ""], ["category", "metric"]], noise_header), axis=1)
    csv_table = np.concatenate((header, body), axis=0)
    csv_list = [row.tolist() for row in csv_table]
    csv_string = "\n".join([",".join(row) for row in csv_list])
    with open(os.path.join(results_root, "scores_summary.csv"), "w+") as f:
        f.write(csv_string)


def train_real_model(real_dataset, real_dataset_path, training_parameters, save_to=None):
    if save_to is None:
        save_to = os.path.join("results_%s" % real_dataset, "results_%s" % real_dataset)

    args = " ".join(["%s %s" % (key, training_parameters[key])
                     for key in training_parameters.keys()])
    command = "python train_and_validate.py -d %s -p %s %s --save_to %s" % \
              (real_dataset, real_dataset_path, args, save_to)
    print("Running: '%s'" % command)
    subprocess.run(command, shell=True)
    print("Done running: '%s'" % command)