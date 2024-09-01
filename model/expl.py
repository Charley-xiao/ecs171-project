import numpy as np
import json
from abc import ABC
import shap
import re
from typing import List, Dict
from lime.lime_text import LimeTextExplainer

class BaseExplainer(ABC):
    def __init__(self, classifier):
        self.classifier = classifier
        label2ind = json.load(open('data/label2ind.json', 'r', encoding='utf-8'))
        self.ind2label = {value: key for key, value in label2ind.items()}
        self.first_three_colors = ['#fead57', '#82aab8', '#92363b']

    def predict_proba(self, texts:list[str]):
        texts = list(texts)
        labels, probabilities = self.classifier.predict(texts, k=len(self.classifier.get_labels()))
        mapped_labels = [[self.ind2label[label] for label in label_list] for label_list in labels]
        mapped_probabilities = [dict(zip(mapped_label_list, prob_list)) for mapped_label_list, prob_list in zip(mapped_labels, probabilities)]
        return np.array([[prob_dict.get(class_name, 0) for class_name in self.ind2label.values()] for prob_dict in mapped_probabilities])

    def explain(self, text:str, top3_author_ids:list):
        """
        Args:
            text: str

        Returns:
            text: str 
                This text is preprocessed, might be different from the user's original input. Please use this text to display the result.
            result: { sentence_index: [(start, end), color(dark grey as default), *(authorID, percentage)] }
        """
        pass

class CustomTokenizer:
    def __init__(self):
        self.vocab = {}
        self.ids_to_tokens = {}
        self.tokenizer_id = 0
    
    def split_rule(self, text: str) -> List[str]:
        sentences = re.split(r'(\!|\.|\?)', text)
        new_sents = []
        for i in range(int(len(sentences)/2)):
            sent = sentences[2*i] + sentences[2*i+1]
            new_sents.append(sent)
        return new_sents
    
    def tokenize(self, text: str) -> List[str]:
        tokens = self.split_rule(text)
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.tokenizer_id
                self.ids_to_tokens[self.tokenizer_id] = token
                self.tokenizer_id += 1
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens[i] for i in ids]
    
    def encode(self, text: str) -> Dict[str, List[int]]:
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)
        offset_mapping = [(m.start(), m.end()) for m in re.finditer('|'.join(map(re.escape, tokens)), text)]
        return {
            'input_ids': input_ids,
            'offset_mapping': offset_mapping
        }
    
    def decode(self, input_ids: List[int]) -> str:
        tokens = self.convert_ids_to_tokens(input_ids)
        return ''.join(tokens)
    
    def __call__(self, text: str) -> Dict[str, List[int]]:
        return self.encode(text)

class ShapExplainer(BaseExplainer):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.tokenizer = CustomTokenizer()
        masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(self.predict_proba, masker, output_names = list(self.ind2label.values()))
        
    def explain(self, text, top3_author_ids):
        if(len(self.tokenizer.split_rule(text)) <= 1):
            return text.replace(' ', ''), None
        
        exp = self.explainer([text]).values[0]
        
        # text = text.replace(' ', '')
        sentences = self.tokenizer.split_rule(text)

        result = {}
        for i in range(len(exp)):
            sentence = sentences[i]
            start_pos = text.find(sentence)
            end_pos = start_pos + len(sentence)

            indexed_list = list(enumerate(exp[i]))
            sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
            top_3_indexes = [(author_id, round(score * 100, 2)) for author_id, score in sorted_indexed_list[:3] if score > 0.001]
            # print(top_3_indexes)

            try:
                rank = top3_author_ids.index(int(top_3_indexes[0][0]))
                color = self.first_three_colors[rank]
            # except ValueError:
            except:
                color = '#212F3D'

            result[i] = [(start_pos, end_pos), color, top_3_indexes]
            
        return text, result

class LimeExplainer(BaseExplainer):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.explainer = LimeTextExplainer(split_expression=self.split_rule, class_names=list(self.ind2label.values()))

    def split_rule(self, text):
        sentences = re.split(r'(。|\!|\.|\?)',text)
        new_sents = []
        for i in range(int(len(sentences)/2)):
            sent = sentences[2*i] + sentences[2*i+1]
            new_sents.append(sent)
        return new_sents

    def explain(self, text, top3_author_ids):
        if(len(self.split_rule(text)) <= 1):
            return text.replace(' ', ''), None
        
        exp = self.explainer.explain_instance(text, self.predict_proba, num_features=1000, top_labels=10)
        exp_map = exp.as_map()
        
        # text = text.replace(' ', '')
        sentences = self.split_rule(text)
        
        newd = {}
        for sent_index, value in exp_map.items():
            newd[sent_index] = [t for t in exp_map[sent_index] if t[1] > 0.01]
        exp_map = newd

        result = {}
        for author_id, tuples in newd.items():
            for sent_index, score in tuples:
                sent_index = int(sent_index)
                sentence = sentences[sent_index]
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence)
                score = round(score * 100, 2)

                if sent_index not in result:
                    result[sent_index] = [(start_pos, end_pos)]
                result[sent_index].append([author_id, score])

        for key, value in result.items():
            positions, *authors_scores = value
            authors_scores = sorted(authors_scores, key=lambda x: x[1], reverse=True)
            try:
                rank = top3_author_ids.index(int(authors_scores[0][0]))
                color = self.first_three_colors[rank]
            # except ValueError:
            except:
                color = '#212F3D'
            result[key] = [positions, color, authors_scores] # sort by score

        for i in range(len(sentences)): # Sentences not similar to any author
            if i not in result:
                sentence = sentences[i]
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence)
                result[i] = [(start_pos, end_pos), '#212F3D', [[-1, 0]]]
        
        result = dict(sorted(result.items(), key=lambda item: item[1][0][0])) # sort by sentence position
        return text, result