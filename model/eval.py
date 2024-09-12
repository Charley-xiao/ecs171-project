import os
import fasttext
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Specify the directory you want to iterate through
directory = 'C:\\VisualStudio\\ecs171-project-main\\model\\'
width = 40
test_file = 'C:\\VisualStudio\\ecs171-project-main\\data\\val_0.19999999999999996.txt' 

def metrics(model,modelname):
    actual_labels = []
    predicted_labels = []
    prob_labels = []

    
    # Read the test file line by line
    with open(test_file, 'r') as f:
        for line in f:
            # Split the line into the label and text
            split_lt = line.strip().split(' ', 1)
              
            # Predict the label for the text
            predicted_label, prob = model.predict(split_lt[1])
        
            # Store the actual and predicted labels
            actual_labels.append(split_lt[0])
            predicted_labels.append(predicted_label[0])
            prob_labels.append(prob)
   
   
    cm = confusion_matrix(actual_labels, predicted_labels) 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{modelname}Confusion Matrix')
    plt.savefig(f'{modelname}.png')  
    
    return classification_report(actual_labels, predicted_labels)


def compare_premade_models():
    #finding number of models for pretty printing
    numberOfModels = 0
    for filename in os.listdir(directory):
        if filename.endswith('.model'):
            numberOfModels+=1

    modelsTested = 1
    eval = ""
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        
        filepath = os.path.join(directory, filename)
        
        progress = (modelsTested / numberOfModels) * 100
            # Create a progress bar
        bar_length = 20
        filled_length = int(round(bar_length * modelsTested / float(numberOfModels)))
        
        
        # Check load/train model
        if filename.endswith('.model'):
            model = fasttext.load_model(filepath)
            progressBars = '#'*filled_length+'-'*(bar_length-filled_length)
            filename = filename.ljust(width)  
            
            modelsTested += 1 
            print(f'Testing: {filename} progress: {progressBars} %{progress}',end='\r')
           
            report = metrics(model, filename)
            result = model.test('../data/train_author_selection.txt')
           
            eval += f"{filename}\nP@1: {result[1]:.3f}  R@1: {result[2]:.3f}\n"
            eval += report
            eval += '\n'
        
    print("\n",end='\r')
    print(eval)
        
compare_premade_models()