import json 
import os
import re
import argparse 
import random 

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_authors', type=int, default=-1, help='Number of authors to include in the dataset.')
args = argparser.parse_args()

DATA_PATH = 'Gutenberg/txt' # Agatha Christie___The Mysterious Affair at Styles.txt
OUTPUT_DATASET_PATH = 'train.txt'
if args.num_authors != -1:
    OUTPUT_DATASET_PATH = f'train_{args.num_authors}.txt'

# Generate a primitive dict that looks like:
# {
#     'Agatha Christie': [{'title': 'The Mysterious Affair at Styles', 'text': '...'}, ...],
#     ...
# }

primitive_dict = {}
label2ind = {}
author_cnt = 0

for filename in os.listdir(DATA_PATH):
    print(f'Processing {filename}...')
    author, title_with_suffix = filename.split('___')
    title = title_with_suffix[:-4]
    print(f'Author: {author}, Title: {title}')
    with open(os.path.join(DATA_PATH, filename), 'r', encoding='ISO-8859-1') as f:
        text = f.read()
    if author not in primitive_dict:
        primitive_dict[author] = []
        label2ind[author] = f'__label__{author_cnt}'
        author_cnt += 1
    primitive_dict[author].append({'title': title, 'text': text})

if args.num_authors == -1:
    with open('label2ind.json', 'w', encoding='ISO-8859-1') as f:
        json.dump(label2ind, f)
else:
    selected_authors = random.sample(list(primitive_dict.keys()), args.num_authors)
    primitive_dict = {author: primitive_dict[author] for author in selected_authors}
    with open(f'label2ind_{args.num_authors}.json', 'w', encoding='ISO-8859-1') as f:
        json.dump(label2ind, f)

def clean_text(text):
    """
    Things to do:
    1. Lowercase the text
    2. Eliminate newlines
    3. Eliminate multiple spaces
    4. Remove any non-alphanumeric characters except for punctuation
    5. Make extra space around punctuation so that it is tokenized
    6. Remove multiple spaces, ensure that there is only one space between tokens
    7. Strip leading and trailing spaces
    8. Three dots (. . .) should be treated as a single token
    9. Extract a list of sentences, ending with punctuation

    For example:
    "Hello, world!" -> "hello , world !"
    """
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,?!:-; ]', '', text)
    text = re.sub(r'([.,?!:-;])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\. \. \.', '...', text)
    sentences = []
    for sentence in re.split(r'(?<=[.?!])', text):
        sentence = sentence.strip()
        if sentence and len(sentence) > 3:
            sentences.append(sentence)
    return sentences

def statistics(primitive_dict):
    print(f'Number of authors: {len(primitive_dict)}')
    print(f'Number of books: {sum(len(primitive_dict[author]) for author in primitive_dict)}')

statistics(primitive_dict)

with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
    for author in primitive_dict:
        for book in primitive_dict[author]:
            sentences = clean_text(book['text'])
            for sentence in sentences:
                f.write(f'{label2ind[author]} {sentence}\n')

print(f'Dataset written to {OUTPUT_DATASET_PATH}.')
