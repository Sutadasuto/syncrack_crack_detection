# syncrack_crack_detection
This repository contains the code used to obtain the results presented in:
```
@conference{visapp22,
  author={Rodrigo Rill{-}García and Eva Dokladalova and Petr Dokládal.},
  title={Syncrack: Improving Pavement and Concrete Crack Detection through Synthetic Data Generation},
  booktitle={Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP},
  year={2022},
  pages={147-158},
  publisher={SciTePress},
  organization={INSTICC},
  doi={10.5220/0010837300003124},
  isbn={978-989-758-555-5},
}
```

It is based on the Syncrack generator, a tool aimed to create synthetic pavement/concrete images with accurate annotations for crack detection.
The most recent version of Syncrack is available at https://github.com/Sutadasuto/syncrack_generator (you must download that repository to use this one).

## How to use
### Pre-requisites
This repository was developed for Python 3.7.6 and tensorflow-gpu. To install the additional required dependencies, we provide an "environment.yml" to clone the conda environment used during our experiments.

### How to use
First, download the Syncrack generator repository in the root folder of this repository. 
Do the same with our pruned version of CFD: https://drive.google.com/file/d/1pTZpiIrqWiUocRsULBRqznEtXrwWYVpQ/view?usp=sharing (uncompress the zip file).
Then, simply run:
```
python run_experiments.py
```

This script will generate the versions of Syncrack used in the paper along with their different noisy versions. General scores will be calculated and saved for these datasets.

Afterwards, U-VGG19 will be trained on all these datasets and the predictions/scores/comparative images will be saved.

Next, U-VGG19 will be trained with CFD and the predictions/scores/comparative images will be saved.

Finally, the models trained on synthetic data with accurate annotations will be used to make predictions on the CFD's validation split. The predictions/scores/comparative images will be saved. These results, as shown in the paper, are available at https://drive.google.com/drive/folders/1aB22te1RQTab74PEXoCWw3Ed73WjGNQC?usp=sharing

While the Syncrack generated datasets should be always the same (fixed random seed), notice that the stochastic methods used to train the neural networks with a GPU will create different models each time the training is called (thus the results you obtain may differ from the ones in the paper).
