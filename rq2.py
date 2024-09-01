"""
RQ2: Comparison between SHAP and LIME.

Consistency: 
    Measure the consistency of explanations across different instances and models 
    (e.g., train the model multiple times with different random seeds and compare the explanations).
Computational Efficiency: 
    Record the time taken and computational resources used to generate explanations with both SHAP and LIME.
"""
from model import *
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import json
import os
import shap

argparser = argparse.ArgumentParser()
argparser.add_argument('--train', type=str, default='data/train.txt', help='Path to the training set.')
argparser.add_argument('--label2ind', type=str, default='data/label2ind.json', help='Path to the label2ind file.')
argparser.add_argument('--existing_model', type=str, default=None, help='Path to the existing model.')
args = argparser.parse_args()

def efficiency(classifier, n=10):
    """
    Record the time taken and computational resources used to generate explanations with both SHAP and LIME.
    """
    label2ind = json.load(open(args.label2ind, 'r', encoding='ISO-8859-1'))

    with open(args.train, 'r', encoding='ISO-8859-1') as f:
        data = f.readlines()

    shap_times = []
    lime_times = []
    
    for i in tqdm(range(n)):
        shap_explainer = ShapExplainer(classifier)
        lime_explainer = LimeExplainer(classifier)
        
        k = random.randint(0, len(data) // 1000)
        lines_list = random.sample(data, k)
        line = ' '.join([' '.join(line.split('__label__')[1].split(' ')[1:]).strip() for line in lines_list])
        token_cnt = len(line.split(' '))

        # line = "Hello , this is a test sentence . I am trying to see how long it takes to explain this sentence . I hope it is not too long . I am trying to make it long enough to be interesting ."
        print(f"Explaining instance:\n {line}")
        result_model_list = predict(line, classifier, args.label2ind)
        result_model_list.sort(key=lambda x: x[2], reverse=True)
        top3_author_ids = [int(result_model[0]) for result_model in result_model_list][:3]
        print(f"Top 3 authors: {top3_author_ids}")
        start = time.time()
        shap_result = shap_explainer.explain(line, top3_author_ids)
        end = time.time()
        print(f"SHAP result: \n{shap_result[0]}\n{shap_result[1]}")
        shap_time = end - start
        start = time.time()
        lime_result = lime_explainer.explain(line, top3_author_ids)
        end = time.time()
        print(f"LIME result: \n{lime_result[0]}\n{lime_result[1]}")
        lime_time = end - start
        shap_times.append((token_cnt, shap_time))
        lime_times.append((token_cnt, lime_time))
        print(f"SHAP time: {shap_time} seconds for {token_cnt} tokens")
        print(f"LIME time: {lime_time} seconds for {token_cnt} tokens")

    shap_times = np.array(shap_times)
    lime_times = np.array(lime_times)
    plt.scatter(shap_times[:, 0], shap_times[:, 1], label='SHAP')
    plt.scatter(lime_times[:, 0], lime_times[:, 1], label='LIME')
    plt.xlabel('Number of tokens')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if args.existing_model is None:
        classifier = fit(args.train)
    else:
        classifier = load_model(args.existing_model)
    efficiency(classifier, n=10)
        


        
