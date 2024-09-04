# Explainable Writing Style Detection with User Interface

## Setup

As you may know, Facebook silently dropped any future support for fastText and it is very important to install the correct version of all the dependencies. To make sure that the project runs correctly, it is strongly recommended to open a new virtual environment and install the dependencies from the `requirements.txt` file.

```bash
conda create -n <your_env_name> python=3.9
pip install -r requirements.txt
```

## Dataset

### Full Dataset

Full dataset is available at [this link](https://drive.google.com/file/d/1OYc1ZZzOeyFWTkMWj3lwfqgMy72B9YVx/view?usp=sharing), which contains a total of 142 authors.

After unzipping the file, you will find two files:

- `label2ind.json`: a json file that maps the author name to an index.
- `train.txt`: a text file that contains the training data. (Approx. 1.2GB)

Please put these two files in the `data` folder.

### Partial Dataset with 10 Authors

Upon evaluation, we found that the full dataset is too large for the current implementation, which also makes it difficult to train the model. Therefore, we decided to use a subset of the full dataset for the current implementation.

Partial dataset is available at [this link](https://drive.google.com/file/d/1RKZTOsffCXKvn8_qzABmw468L4ZbmcGp/view?usp=sharing), which contains a total of manually selected 10 authors, including:

- Charles Dickens,
- Agatha Christie,
- Jane Austen,
- Mark Twain,
- O Henry,
- Oscar Wilde,
- P G Wodehouse,
- Walt Whitman,
- Winston Churchill, and 
- Zane Grey.

### Generate Dataset from Scratch

If you wish to generate the dataset from scratch, here are the steps:

1. Download the raw data from [a preprocessed Gutenberg dataset](https://drive.google.com/file/d/1i8eeP79dN2TwIK7H4qr_Y-ji1cB19SMU/view).
2. Unzip the file and put the `Gutenberg` folder under the `data` folder.
3. In the `data` folder, run the following command to generate the dataset:

```bash
python preprocess.py --num_authors <number_of_authors_you_want>
```

This will generate the dataset with the specified number of authors **randomly**. The generated dataset will be saved in the `data` folder.

If you wish to generate the dataset with specific authors, please modify the `manually_selected_authors` variable in the `preprocess.py` file, and run:

```bash
python preprocess.py --enable_author_selection
```

If you wish to split the dataset into training and testing sets, you can run the following command

```bash
python splitdataset.py --train_split <train_split_ratio>
```

after you properly modify the variable `ORIGINAL_DATASET_PATH` in the `splitdataset.py` file.

## Model

### Training

To train the model, you can run the following command under the `model` folder:

```bash
python train.py --train <path_to_the_training_data> --type <basic_or_autotune> --test <path_to_the_testing_data> --val <path_to_the_validation_data> --model <path_to_save_the_model> --label2ind <path_to_the_label2ind_file>
```

### Use Pretrained Model

We have provided a pretrained model at [this link](https://drive.google.com/file/d/1Vq7f2tPF6SBCA3jbioRHjsVBDSh2Nadk/view?usp=sharing). You can download the model and put it under the `model` folder.

## User Interface

The user interface is implemented using Tornado, a Python web framework. To run the user interface, you can run the following command under the root directory:

```bash
python index.py --model <path_to_the_model> --label2ind <path_to_the_label2ind_file>
```

Then, you can access the user interface by visiting `http://localhost:9263` in your web browser.

> [!IMPORTANT]  
> To run the server successfully, your folder structure should at least look like this:
> ```
> .
> ├── data
> │   └── label2ind.json
> ├── model
> │   ├── __init__.py
> │   ├── expl.py
> │   ├── train.py
> │   └── some_model_file
> ├── templates
> │   ├── index.html
> │   └── result.html
> └── index.py

## Acknowledgements

This project is a part of the course project for the course ECS 171 Machine Learning SS2 2024 at University of California, Davis. The project is done by the following members (in alphabetical order):

- [Abdullah Al Rawi](https://github.com/Abdullah-alrawi2002)
- [Andrew Hoang](https://github.com/andrewh965)
- [Ching Hao Chang](https://github.com/justin00195)
- [Qiwen Xiao](https://github.com/Charley-xiao)
- [Weifeng Liu](https://github.com/weiL593)

We would like to thank the following open-source contributors for their work.

- Some portions of the code in this project are adapted from Ankie Fan's project [testurtext-algo](https://github.com/AnkieFan/testurtext-algo/tree/main).
- The dataset used in this project is based on the [Gutenberg dataset](https://shibamoulilahiri.github.io/gutenberg_dataset.html) provided by Lahiri et al. (2014).
- The implementation of the model is based on the [fastText](https://fasttext.cc/) library provided by Facebook Research.
- The user interface is implemented using the [Tornado](https://www.tornadoweb.org/en/stable/) web framework.
