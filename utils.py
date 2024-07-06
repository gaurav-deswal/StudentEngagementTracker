import torchvision
from torch.utils.data import DataLoader
import torch
import datasets
import transforms
import numpy as np
import os
from tqdm import tqdm
import pickle
import torchvision.transforms as transform1
def generate_dataloader(batch_size, csv, root):
    dataset = datasets.VideoDataset(csv,
                                    root,
                                    transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      )


def get_dataloader(batch_size, csv_train, root_train, csv_test, root_test):
    return {
        'train': generate_dataloader(batch_size, csv_train, root_train),
        'test': generate_dataloader(batch_size, csv_test, root_test)}



def generate_dataloader_weighted(batch_size, csv, root,weights):
    
    class_distribution = weights

    # Step 1: Inverse Class Frequency
    inverse_class_freq = 1.0 / class_distribution

    # Step 2: Normalize Weights
    normalized_weights = inverse_class_freq / np.sum(inverse_class_freq)

    # Step 3: Class Re-weighting

    # You can adjust this factor based on your preference for penalization
    reweighting_factor = 0.5  # Adjust as needed
    samples_weight = list(normalized_weights ** reweighting_factor)
    # samples_weight = [0.6012333 , 0.08016444 ,0.1336074 , 0.18499486]
    print("Weights for WeightedRandomSampler:", samples_weight)

    dataset = datasets.VideoDataset(csv,
                                    root,
                                    transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor()]))
    
    pickle_file = "weights_mod-1.pkl"

    if os.path.exists(pickle_file):
        # If pickle file exists, load it directly
        with open(pickle_file, 'rb') as f:
            weights_list = pickle.load(f)
    else:
        # If pickle file doesn't exist, calculate weights and save to pickle file
        weights_list = []

        for instance in tqdm(dataset):
            label = instance[1]
            weight = samples_weight[label]
            weights_list.append(weight)

        # Save weights to pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(weights_list, f)


    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_list, len(weights_list),replacement=True)
    print('batchsize_ ',batch_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      )


def get_dataloader_weighted(batch_size, csv_train, root_train, csv_test, root_test):
    weights = np.array([  34,  214, 2649, 2585])
    return {
        'train': generate_dataloader_weighted(batch_size, csv_train, root_train,weights),
        'test': generate_dataloader(batch_size, csv_test, root_test)}


def generate_dataloader_aug_weighted(batch_size, csv, root,weights):


# sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    
    
    dataset = datasets.VideoDataset(csv,
                                    root,
                                    transform=torchvision.transforms.Compose([transforms.VideoFolderPathToTensor(), transform1.RandomHorizontalFlip(),transform1.RandomRotation(degrees=10),transform1.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]))
    

    print("dataset size",len(dataset))
    # class_distribution = weights

    # # Step 1: Inverse Class Frequency
    # inverse_class_freq = 1.0 / class_distribution

    # # Step 2: Normalize Weights
    # normalized_weights = inverse_class_freq / np.sum(inverse_class_freq)

    # # Step 3: Class Re-weighting

    # # You can adjust this factor based on your preference for penalization
    # reweighting_factor = 0.5  # Adjust as needed
    # samples_weight = list(normalized_weights ** reweighting_factor)
    # print(samples_weight)
    # sum_weights = sum(samples_weight)

    # # Step 2: Normalize each sample weight
    # samples_weight = [weight / sum_weights for weight in samples_weight]

    # pickle_file = "weights_mod_aug_0.5.pkl"
    # print("generate_dataloader_aug_weighted() ",samples_weight)
    # if os.path.exists(pickle_file):
    #     # If pickle file exists, load it directly
    #     with open(pickle_file, 'rb') as f:
    #         weights_list = pickle.load(f)
    # else:
    #     # If pickle file doesn't exist, calculate weights and save to pickle file
    #     weights_list = []

    #     for instance in tqdm(dataset):
    #         label = instance[1]
    #         weight = samples_weight[label]
    #         weights_list.append(weight)

    #     # Save weights to pickle file
    #     with open(pickle_file, 'wb') as f:
    #         pickle.dump(weights_list, f)

    # print("weights_list ",len(weights_list)) 
    # print(weights_list)

    # sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    return DataLoader(dataset,
                      batch_size=batch_size,
                    #   sampler=sampler,
                     num_workers=2 )

def get_dataloader_aug_weighted(batch_size, csv_train, root_train, csv_test, root_test):
    weights = np.array([  34,  214, 2649, 2585])
    return {
        'train': generate_dataloader_aug_weighted(batch_size, csv_train, root_train,weights),
        'test': generate_dataloader(batch_size, csv_test, root_test)}