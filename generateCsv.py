import os
import csv

os.chdir(r"F:\RESTCN_CODE")

# Define the root directory where your DAiSEE data is stored
root_dir = 'Dataset'
# Define where to save the CSV files
output_dir = 'csv'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the phases and their corresponding frame directories
phases = [('Validation', 'Validation_Frames')]#[('Train', 'Train_Frames'), ('Test', 'Test_Frames')] # [('Train', 'Train_Frames'), ('Test', 'Test_Frames'), ('Validation', 'Validation_Frames')]

# Load labels from AllLabels.csv - We are picking label as the value of Engagement column for every video. Other columns are not considered
labels_path = os.path.join(root_dir, 'Labels', 'AllLabels.csv') # Labels for every video are stored in original Dataset's AllLabels.csv file
labels = {}
with open(labels_path, mode='r') as label_file:
    reader = csv.reader(label_file)
    next(reader, None)  # Skip the header row
    for row in reader:
        video_name = row[0].split('.')[0]  # Assuming video_name is unique and can be used as a key. I have removed the extensions (.avi, .mp4) from video names while storing it.
        label = row[2]  # Label is in the third column
        labels[video_name] = label

# print(f"Labels: {dict(sorted(labels.items()))}") # Print Labels dictionary sorted by video names.
# OUTPUT: Labels: {'1100011002': '2', '1100011003': '2', '1100011004': '3', '1100011005': '3', '1100011006': '3', '1100011007': '2', '1100011008': '3', '1100011009': '2', '1100011010': '3', '1100011011': '3', '1100011012': '2', '1100011013': '3' ...}

# Function to get a label for a given video name
def get_label_for_video(video_name):
    if video_name not in labels:
        labels_filename = labels_path.split('\\')[-1]
        # raise ValueError(f"Label for video '{video_name}' not found in {labels_filename} file.")
        print(f"{video_name} not found in AllLabels.csv file")
        return None
    return labels[video_name]


# Iterate over each phase and generate the CSV files
for phase, frame_dir_name in phases:
    csv_file_path = os.path.join(output_dir, f'{phase.lower()}.csv')
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['path', 'label'])  # Header

        # Path to the frame directory for the current phase
        phase_frame_dir = os.path.join(root_dir, frame_dir_name) # Dataset\Train_Frames
        # Iterate over all subjects in the phase
        for subject in os.listdir(phase_frame_dir):
            subject_path = os.path.join(phase_frame_dir, subject) # Dataset\Train_Frames\110001
            # Iterate over all videos for the subject
            for video in os.listdir(subject_path):
                video_path = os.path.join(subject_path, video) # Dataset\Train_Frames\110001\1100011002
                # Get the label for the video
                video_name = video_path.split('\\')[-1]
                label = get_label_for_video(video_name)
                if label is None: # If for a particular video, we don't have the true label available in our csv file, skip that video 
                    continue
                # Write the video path and label to the CSV
                writer.writerow([video_path, label])
    print(f"{phase}.csv file generated successfully.")    
print("All CSV files generated successfully.")