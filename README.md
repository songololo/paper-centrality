# Paper on Centralities and Normalisation

Paper on centralities and normalisation methods

## Installation

Clone this repository to a local working folder.

The PDM package manager is recommended and can be installed on mac per `brew install pdm`.

Packages can then be installed into a virtual environment per `pdm install`.

If using an IDE the `.venv` should be detected automatically by IDEs such as vscode.

## Data

The dataset is prepared by running the [Madrid UA Dataset](https://github.com/songololo/madrid-ua-dataset).

Generate a copy of the dataset called `dataset.gpkg` and add it to this repository in a folder called `temp`. The `temp` folder is ignored per `.gitignore` but is required for the dataset to be found by the Python scripts.

## Processing

It is recommended to use an IDE such as vscode to run the cell blocks directly. Cell blocks are used instead of Jupyter notebooks because the latter can cause issues and bloat for code repositories.
