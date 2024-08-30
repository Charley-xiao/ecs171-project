import fasttext
from sklearn.model_selection import StratifiedKFold
from train import fit 
import os


def removeFile(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except PermissionError:
        print(f"Permission denied: unable to delete {file_path}.")
    except Exception as e:
        print(f"Error: {e}")

#might not be neccessary to include classifier path since we are retraining model every iteration
def cross_validation(classifier_path='./classifier.model', fold=4, dataSet='../data/train_author_selection.txt'):
    
    #dont really need will delete later
    classifier = fasttext.load_model(classifier_path)
    
    with open(dataSet,'r') as file:
        lines=file.readlines()
    
    labels = []
    texts  = []
    
    #extract labels and text from file. Required for skf.split
    for line in lines:
        label,text = line.split(' ',1)
        labels.append(label)
        texts.append(text.strip())
        
    #skf splits dataset into train and test
    skf = StratifiedKFold(n_splits=fold, shuffle = True)
    accuracies = []  #will delete l8r
    report     = []  #records fold acc
    
    i = 1  #for printing asthetic
    for train_index, valid_index in skf.split(texts, labels):
        train_data = [(labels[i], texts[i]) for i in train_index]
        valid_data = [(labels[i], texts[i]) for i in valid_index]
        
        #write temp files for fasttext training
        #fasttext does not have continuing training on model
            #possible alternative: save hyperparams and use on next fold?
        
         # Save the training data to a file
        with open('./crossTempTr.txt', 'w') as f:
            for label, text in train_data:
                f.write(f"{label} {text}\n")
        
        # Save the validation data to a file
        with open('./crossTempV.txt', 'w') as f:
            for label, text in valid_data:
                f.write(f"{label} {text}\n")
            
        classifier = fasttext.train_supervised(input = './crossTempTr.txt')
        accuracy = classifier.test('./crossTempV.txt') 

        #remove temp files
        removeFile('./crossTempTr.txt')
        removeFile('./crossTempV.txt')
        accuracies.append(accuracy[1])
        report.append(f"Fold {i}: {accuracy[1]:.4f}")
        i+=1

    mean_accuracy = sum(accuracies) / len(accuracies)
    print(report)
    print(f"Mean Accuracy: {mean_accuracy}")
    

cross_validation(fold = 4)
#8=D