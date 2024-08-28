## Dataset

### Full Dataset

Full dataset is available at [this link](https://drive.google.com/file/d/1OYc1ZZzOeyFWTkMWj3lwfqgMy72B9YVx/view?usp=sharing), which contains a total of 142 authors.

After unzipping the file, you will find two files:

- `label2ind.json`: a json file that maps the author name to an index.
- `train.txt`: a text file that contains the training data. (Approx. 1.2GB)

Please put these two files in the `data` folder.

### Partial Dataset with 10 Authors

Upon evaluation, we found that the full dataset is too large for the current implementation, which also makes it difficult to train the model. Therefore, we decided to use a subset of the full dataset for the current implementation.

Partial dataset is available at [this link](), which contains a total of manually selected 10 authors. **(Temporarily unavailable)**

### Generate Dataset from Scratch

If you wish to generate the dataset from scratch, here are the steps:

1. Download the raw data from [a preprocessed Gutenberg dataset](https://drive.google.com/file/d/1i8eeP79dN2TwIK7H4qr_Y-ji1cB19SMU/view).
2. Unzip the file and put the `Gutenberg` folder under the `data` folder.
3. In the `data` folder, run the following command to generate the dataset:

```bash
python preprocess.py --num_authors <number of authors you want>
```

This will generate the dataset with the specified number of authors **randomly**. The generated dataset will be saved in the `data` folder.

If you wish to split the dataset into training and testing sets, you can run the following command

```bash
python splitdataset.py --train_split <train split ratio>
```

after you properly modify the variable `ORIGINAL_DATASET_PATH` in the `splitdataset.py` file.

## Model

### Training

To train the model, you can run the following command under the `model` folder:

```bash
python train.py --train <path to the training data> --type <basic or autotune> --test <path to the testing data> --val <path to the validation data> --model <path to save the model>
```

## User Interface

The user interface is implemented using Tornado, a Python web framework. To run the user interface, you can run the following command under the root directory:

```bash
python index.py
```

Then, you can access the user interface by visiting `http://localhost:9263` in your web browser.

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
