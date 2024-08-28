import fasttext
import json

def fit(training_set_path, type='basic', validation_set_path=None):
    if type == 'basic':
        return fasttext.train_supervised(
            input=training_set_path, 
            wordNgrams=2, 
            epoch=20, 
            lr=0.5, 
            dim=300
        ) 
    elif type == 'autotune':
        assert validation_set_path is not None, 'Validation set path is required for autotune.'
        return fasttext.train_supervised(
            input=training_set_path, 
            autotuneValidationFile=validation_set_path, 
            autotuneModelSize='3600M', 
            autotuneDuration=3600, 
            autotuneMetric='f1'
        )
    else:
        raise ValueError(f'Invalid type: {type}. Use "basic" or "autotune".')

def predict(text, classifier, k=3):
    """
    Returns:
        list: [[id, author name, percentage], ...]
    """
    labels, probs = classifier.predict(text, k=k)
    label2ind = json.load(open('data/label2ind.json', 'r', encoding='utf-8'))
    result = []
    for i in range(k):
        id = labels[i][9:]
        name = next(key for key, val in label2ind.items() if val == labels[i])
        result.append([id,name, round(probs[i] * 100, 2)])
    return result

def load_model(model_path):
    return fasttext.load_model(model_path)

if __name__ == '__main__':
    import argparse 

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', type=str, default='../data/train.txt', help='Path to the training set.')
    argparser.add_argument('--type', type=str, choices=['basic', 'autotune'], default='basic', help='Type of training. Default: basic')
    argparser.add_argument('--val', type=str, default=None, help='Path to the validation set. Required for autotune.')
    argparser.add_argument('--test', type=str, default=None, help='Path to the test set. Default: training set.')
    argparser.add_argument('--model', type=str, default='classifier.model', help='Path to save the model. Default: classifier.model')
    args = argparser.parse_args()
    if args.test is None:
        args.test = args.train

    classifier = fit(args.train, type=args.type, validation_set_path=args.val)
    result = classifier.test(args.test)
    print("Model results: ")
    print('P@1:', result[1])
    print('R@1:', result[2])
    print('Number of examples:', result[0])
    classifier.save_model(args.model)

# Read 242M words
# Number of words:  1009665
# Number of labels: 142
# Progress: 100.0% words/sec/thread:  394318 lr:  0.000000 avg.loss:  4.007396 ETA:   0h 0m 0s
# Model results: 
# P@1: 0.1808300395256917
# R@1: 0.1808300395256917
# Number of examples: 3036