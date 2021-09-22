import os

from data import generate_datasets, generate_dataset_summary, generate_plots, train_synthetic_model, \
    create_syncthetic_training_summary, train_real_model

############################################## Solely synthetic data results ##########################################

# Folder in which the generated datasets will be saved
datasets_folder = "/media/shared_storage/datasets/syncrack_parametrized"
# Path to Syncrack generator
syncrack_generator = "/home/sutadasuto/Dropbox/ESIEE/PhD/Year_2/Software/syncrack_generator"
# The different datasets to create. It is a dict with dataset name as key, and the arguments to pass to the syncrack
# generator as dicts too
category_args_dict = {
    "easy": {"-bas": "6.0", "-cac": "0.5"},
    "normal": {"-bas": "3.0", "-cac": "0.7"},
    "hard": {"-bas": "1.5", "-cac": "0.7"}
}
synthetic_training_parameters = {
    "-da": "True"
}
# The different noise levels (percentage from 0 to 100 as integer) to introduce to each of the generated datasets
noise_levels = [0, 25, 50, 75, 100]
supervised_metrics = ["dice_score_coefficient", "precision", "recall"]
unsupervised_metrics = ["crack_region_entropy", "crack_region_second_order_entropy", "kolmogorov-smirnov_alpha5"]

# Step flags: decide whether to do or not each of these steps
generate_datasets_flag = True
create_dataset_summaries_flag = True
create_plots_flag = True
train_networks_flag = True
create_training_summary_flag = True

#######################################################################################################################

############################################## Real and synthetic data results ########################################

training_parameters = {

}

real_dataset = "cfd"
real_dataset_path = "/media/shared_storage/datasets/CrackForestDatasetPruned"

train_real_network_flag = True
validate_synthetic_on_real_flag = True
validate_noisy_models = False

#######################################################################################################################

categories = category_args_dict.keys()
# Ensure to always have a clean annotations (noise percentage level 0)
noise_levels.sort()
if not noise_levels[0] == 0:
    noise_levels = [0] + noise_levels

# Generate the datasets
for category in categories:
    if generate_datasets_flag:
        generate_datasets(syncrack_generator, category_args_dict[category], datasets_folder, category, noise_levels)
    if create_dataset_summaries_flag:
        generate_dataset_summary(datasets_folder, category, noise_levels)

    if create_plots_flag:
        generate_plots(supervised_metrics, unsupervised_metrics, noise_levels, datasets_folder, category, zoom=True,
                       order=False)

    with open(os.path.join(datasets_folder, "parameters.txt"), 'w+') as f:
        f.write("\n".join([str(category_args_dict), str(noise_levels)]))

results_root = "results_synthetic"
if train_networks_flag:
    #  Train a neural network per category and noise level on the synthetic images
    for category in categories:
        for noise_level in noise_levels:
            train_synthetic_model(datasets_folder, results_root, synthetic_training_parameters, noise_level, category)

if create_training_summary_flag:
    # Create summary of all the results
    create_syncthetic_training_summary(results_root, list(categories), noise_levels)


if train_real_network_flag:
    # Train a network using only the provided real-images dataset
    train_real_model(real_dataset, real_dataset_path, training_parameters)


if validate_synthetic_on_real_flag:
    # Validate the models
    noisy_models = noise_levels if validate_noisy_models else [0]
    validation_args = {
        "-e": 0,
        "-w": None
    }
    for category in categories:
        for noisy_model in noisy_models:
            validation_args["-w"] = os.path.join(results_root, "results_%s" % category, "%s_percent_noise" % noisy_model,
                                                 "results_training_min_val_loss", "uvgg19_best.hdf5")
            save_to = os.path.join("results_%s" % real_dataset, "results_%s_%s_percent_noise" % (category, noisy_model))
            train_real_model(real_dataset, real_dataset_path, validation_args, save_to=save_to)
