import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
from distutils.util import strtobool
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as keras_metrics
from tensorflow.keras.models import model_from_json
from json.decoder import JSONDecodeError

from callbacks_and_losses import custom_losses
import data

from callbacks_and_losses.custom_calllbacks import EarlyStoppingAtMinValLoss, ReduceLROnPlateau, TensorBoard
from models.available_models import get_models_dict

# Used for memory error in RTX2070
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

models_dict = get_models_dict()


def main(args):
    start = datetime.now().strftime("%d-%m-%Y_%H.%M")
    results_dir = "results_%s" % start if args.save_to is None else args.save_to
    results_train_dir = os.path.join(results_dir, "results_training")
    results_train_min_loss_dir = results_train_dir + "_min_val_loss"
    results_validation_dir = os.path.join(results_dir, "results_validation")
    results_validation_min_loss_dir = results_validation_dir + "_min_val_loss"

    # Here we find to paths to all images from the selected datasets
    paths = data.create_image_paths(args.dataset_names, args.dataset_paths)
    if args.clean_dataset_paths is not None:
        clean_paths = data.create_image_paths(args.dataset_names, args.clean_dataset_paths)
        results_clean_validation_min_loss_dir = os.path.join(results_dir, "results_clean")
    # Data is split into training data and validation data. A custom seed is used for reproducibility.
    n_training_images = int(args.training_split_ratio * paths.shape[1])
    np.random.seed(0)
    order = np.array([i for i in range(paths.shape[-1])])
    np.random.shuffle(order)
    paths = paths[:, order]
    training_paths = paths[:, :n_training_images]
    validation_paths = paths[:, n_training_images:]
    if args.clean_dataset_paths is not None:
        clean_paths = clean_paths[:, order]
        clean_validation_paths = clean_paths[:, n_training_images:]

    # If asked by the user, save the paths of the validation split (useful to validate later)
    if args.save_validation_paths:
        with open("validation_paths.txt", "w+") as file:
            file.write("\n".join([";".join(paths) for paths in validation_paths.transpose()]))
            print("Validation paths saved to 'validation_paths.txt'")

    # If asked by the user, save the paths of the validation split (useful to repeat later)
    if args.save_training_paths:
        with open("training_paths.txt", "w+") as file:
            file.write("\n".join([";".join(paths) for paths in training_paths.transpose()]))
            print("Training paths saved to 'validation_paths.txt'")

    # As input images can be of different sizes, here we calculate the total number of patches used for training.
    print("Calculating the total number of samples after cropping and data augmentatiton. "
          "This may take a while, don't worry.")
    # We don't resize images for training, but provide image patches of reduced size for memory saving
    # An image is turned into this size patches in a chess-board-like approach
    input_size = args.training_crop_size
    n_train_samples = next(data.train_image_generator(training_paths, input_size, args.batch_size,
                                                      count_samples_mode=True, rgb_preprocessor=None,
                                                      data_augmentation=False))  # Don't preprocess images with DA

    while True:
        tf.keras.backend.clear_session()
        print("\nCreating and compiling model.")
        input_size = (None, None)
        # Load model from JSON file if file path was provided...
        if os.path.exists(args.model):
            try:
                with open(args.model, 'r') as f:
                    json = f.read()
                model = model_from_json(json)
                args.model = os.path.splitext(os.path.split(args.model)[-1])[0]
            except JSONDecodeError:
                raise ValueError(
                    "JSON decode error found. File path %s exists but could not be decoded; verify if JSON encoding was "
                    "performed properly." % args.model)
        # ...Otherwise, create model from this project by using a proper key name
        else:
            model = models_dict[args.model]((input_size[0], input_size[1], 1))
        try:
            # Model name should match with the name of a model from
            # https://www.tensorflow.org/api_docs/python/tf/keras/applications/
            # This assumes you used a model with RGB inputs as the first part of your model,
            # therefore your input data should be preprocessed with the corresponding
            # 'preprocess_input' function
            m = importlib.import_module('tensorflow.keras.applications.%s' % model.name)
            rgb_preprocessor = getattr(m, "preprocess_input")
        except ModuleNotFoundError:
            rgb_preprocessor = None

        # We don't resize images for training, but provide image patches of reduced size for memory saving
        # An image is turned into this size patches in a chess-board-like approach
        input_size = args.training_crop_size

        # Model is compiled so it can be trained
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=custom_losses.bce_dsc_loss(args.alpha),
                      metrics=[custom_losses.dice_coef, 'binary_crossentropy',
                               keras_metrics.Precision(), keras_metrics.Recall()])

        # For fine tuning, one can provide previous weights
        if args.pretrained_weights:
            model.load_weights(args.pretrained_weights)

        print("\nProceeding to train.")

        # A customized early stopping callback. At each epoch end, the callback will test the current weights on the
        # validation set (using whole images instead of patches) and stop the training if the minimum validation loss
        # hasn't improved over the last 'patience' epochs.
        es = EarlyStoppingAtMinValLoss(validation_paths, file_path='%s_best.hdf5' % args.model, patience=args.patience,
                                       rgb_preprocessor=rgb_preprocessor)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs")

        # Training begins. Note that the train image generator can use or not data augmentation through the parsed
        # argument 'use_da'
        print("Start!")
        try:
            history = model.fit(x=data.train_image_generator(training_paths, input_size, args.batch_size,
                                                             rgb_preprocessor=rgb_preprocessor,
                                                             data_augmentation=args.use_da),
                                epochs=args.epochs,
                                verbose=1, callbacks=[es, reduce_lr, tensorboard],
                                steps_per_epoch=n_train_samples // args.batch_size)
        except KeyboardInterrupt:
            # This assumes a user keyboard interruption, as a hard early stop
            print("Training interrupted. Only the best model will be saved.")
            # Load the weights from the epoch with the minimum validation loss
            model.load_weights('%s_best.hdf5' % args.model)
            args.epochs = 0

        if not es.bad_ending:
            print("Finished!")
            break
        else:
            print("Failed convergence. A new model will be created and trained.")
            del model

    # Evaluate using the best validation weights.
    if args.epochs > 1:
        # Load results using the min val loss epoch's weights
        model.load_weights('%s_best.hdf5' % args.model)
    print("Evaluating the model with minimum validation loss...")
    print("On training paths...")
    data.evaluate_model_on_paths(model, training_paths, results_train_min_loss_dir, args)
    print("On validation paths...")
    data.evaluate_model_on_paths(model, validation_paths, results_validation_min_loss_dir, args)
    if args.clean_dataset_paths is not None:
        print("On clean validation paths...")
        data.evaluate_model_on_paths(model, clean_validation_paths, results_clean_validation_min_loss_dir, args)
    # Move the best weights file if trained by at least 1 epoch
    if args.epochs > 0:
        os.replace('%s_best.hdf5' % args.model, os.path.join(results_train_min_loss_dir, '%s_best.hdf5' % args.model))

    # If trained more than one epoch, save the training history as csv and plot it
    if args.epochs > 1:
        print("\nPlotting training history...")
        import pandas as pd
        pd.DataFrame.from_dict(history.history).to_csv(os.path.join(results_dir, "training_history.csv"), index=False)
        # summarize history for loss
        for key in history.history.keys():
            plt.plot(history.history[key])
        plt.ylim((0.0, 1.0 + args.alpha))
        plt.title('model losses')
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.legend(history.history.keys(), loc='upper left')
        plt.savefig(os.path.join(results_dir, "training_losses.png"))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_names", type=str, nargs="+",
                        help="Must be one of: 'cfd', 'cfd-pruned', 'aigle-rn', 'esar', 'crack500', 'gaps384', "
                             "'cracktree200', 'text'")
    parser.add_argument("-p", "--dataset_paths", type=str, nargs="+",
                        help="Path to the folders or files containing the respective datasets as downloaded from the "
                             "original source.")
    parser.add_argument("-cp", "--clean_dataset_paths", type=str, nargs="+", default=None,
                        help="Path to the folders or files containing the respective datasets as downloaded from the "
                             "original source.")
    parser.add_argument("-m", "--model", type=str, default="uvgg19",
                        help="Network to use. It can be either a name from 'models.available_models.py' or a path to a "
                             "json file.")
    parser.add_argument("-cs", "--training_crop_size", type=int, nargs=2, default=[256, 256],
                        help="For memory efficiency and being able to admit multiple size images,"
                             "subimages are created by cropping original images to this size windows")
    parser.add_argument("-a", "--alpha", type=float, default=3.0,
                        help="Alpha for objective function: BCE_loss + alpha*DICE")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate for Adam optimizer.")
    parser.add_argument("-e", "--epochs", type=int, default=150, help="Number of epochs to train.")
    parser.add_argument("-bs", "--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("-ts", "--training_split_ratio", type=float, default=0.5,
                        help="The ratio between the number of training images and the total number of images.")
    parser.add_argument("--patience", type=int, default=20, help="Early stop patience.")
    parser.add_argument("-da", "--use_da", type=str, default="False", help="If 'True', training will be done using data "
                                                                   "augmentation. If 'False', just raw images will be "
                                                                   "used.")
    parser.add_argument("-w", "--pretrained_weights", type=str, default=None,
                        help="Load previous weights from this location.")
    parser.add_argument("--save_to", type=str, default=None,
                        help="Results will be saved in this location. If not provided, a folder 'results_date_hour'"
                             "will be created for this purpose.")
    parser.add_argument("--save_validation_paths", type=str, default="False", help="If 'True', a text file "
                                                                                   "'validation_paths.txt' containing "
                                                                                   "the paths of the images used "
                                                                                   "for validating will be saved in "
                                                                                   "the project's root.")
    parser.add_argument("--save_training_paths", type=str, default="False", help="If 'True', a text file "
                                                                                   "'training_paths.txt' containing "
                                                                                   "the paths of the images used "
                                                                                   "for training will be saved in "
                                                                                   "the project's root.")

    args_dict = parser.parse_args(args)
    for attribute in args_dict.__dict__.keys():
        if args_dict.__getattribute__(attribute) == "None" or args_dict.__getattribute__(attribute) == ["None"]:
            args_dict.__setattr__(attribute, None)
        if args_dict.__getattribute__(attribute) == "True" or args_dict.__getattribute__(attribute) == "False":
            args_dict.__setattr__(attribute, bool(strtobool(args_dict.__getattribute__(attribute))))
    return args_dict


if __name__ == "__main__":
    args = parse_args()
    main(args)
