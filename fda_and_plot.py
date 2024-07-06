import torch
import os
os.chdir("F:\RESTCN_CODE")
# import cv2
import pandas as pd
from ResTCN import ResTCN
from transforms_modified import VideoFolderPathToTensor


# Function to load the saved model
def load_model(model_path, device):
    model = ResTCN().to(device)
    model.load_state_dict(torch.load(model_path))
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict_class(model, path, device):
    video_transform = VideoFolderPathToTensor(max_len=16)  # Assuming max_len is properly utilized in your transform
    
    # Applying the video transformation
    video_tensor = video_transform(path)  # 'path' should be the directory containing frames of a video
    
    # Ensure the tensor is on the correct device
    video_tensor = video_tensor.to(device)
    
    # Predicting with the model
    with torch.no_grad():
        outputs = model(video_tensor.unsqueeze(0))  # Ensure the input is batched
    torch.cuda.empty_cache()
    predicted = torch.max(outputs, 1)[1].item()
    
    return video_tensor, predicted, outputs

# Initialize lists for storing classified frames' paths
train_lists = [[] for _ in range(4)]  # Four lists for train data
test_lists = [[] for _ in range(4)]  # Four lists for test data

# Load the model
device = torch.device("cuda")  # Assuming you're loading on a CPU; change to "cuda" if using GPU
model_path = "resTCN_mod_model_weighted.pth"
model = load_model(model_path, device)
model.eval()  # Set model to evaluation mode

def process_csv(csv_path, output_lists):
    
    device = torch.device("cuda")
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        video_path = row['path']  # This should be a directory containing frames
        true_label = row['label']  # Ensure your CSV has a 'label' column
        
        video_tensor, predicted_class, model_output  = predict_class(model, video_path, device)
        print(f"Predicted Class: {predicted_class}\tTrue Class: {true_label}")
        if len(output_lists[true_label]) < 10000:
              # Store a tuple of (input tensor, true label, model output)
            output_lists[true_label].append((video_tensor, true_label, model_output))
        # if all(len(lst) == 10 for lst in output_lists):  # Stop if all lists have 5 entries
        #     break
    return
        
# Process train and test CSV files
train_csv = 'csv/train.csv'
test_csv = 'csv/test.csv'

print("Train CSV processing...")
process_csv(train_csv, train_lists)
print(36 * "-")
print("Test CSV processing...")
process_csv(test_csv, test_lists)

# print("Train Lists:", train_lists)
# print("Test Lists:", test_lists)


# APPLYING FDA and plotting train and test datasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def apply_fda_and_plot(data_lists, title):
    # Prepare data and labels
    X = []
    y = []
    for class_index, class_entries in enumerate(data_lists):
        for entry in class_entries:
            # Assuming entry[0] (video_tensor) is already flattened or processed to represent features
            # We might need to adjust here depending on the actual shape and content of entry[0]
            features = entry[0].flatten().cpu().numpy()  # Flatten and convert to numpy
            X.append(features)
            y.append(entry[1])  # true_label

    X = np.array(X)
    y = np.array(y)

    # Applying FDA (LDA here performs a similar role)
    lda = LDA(n_components=2)  # Reduce dimensions to 2
    X_r = lda.fit(X, y).transform(X)

    # Plotting
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    lw = 2
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    plt.figure()
    for color, i, label in zip(colors, [0, 1, 2, 3], labels):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=label)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()

# Example usage:
print("Applying FDA on Train data and plotting it color coded for 4 different classes...")
apply_fda_and_plot(train_lists, "FDA Training Data")
print("FDA applied successfully on train data.")
print("Applying FDA on Test data and plotting it color coded for 4 different classes...")
apply_fda_and_plot(test_lists, "FDA Testing Data")
print("FDA applied successfully on test data.")