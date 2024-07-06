import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import pickle
import numpy as np

from ResTCN import ResTCN
from utils import get_dataloader_aug_weighted

import warnings
warnings.filterwarnings("ignore")

os.chdir(r"F:\RESTCN_CODE")
if __name__=='__main__':
        
    torch.manual_seed(0)
    num_epochs = 25
    batch_size = 6
    lr = .001
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device being used:", device, flush=True)


    # dataloader = get_dataloader(batch_size,
    #                             'train.csv',
    #                             os.path.join(os.getcwd(), 'images_train'),
    #                             'test.csv',
    #                             os.path.join(os.getcwd(), 'images_test'))
    dataloader = get_dataloader_aug_weighted(batch_size,
                                'csv\\train.csv',
                                os.getcwd(),
                                'csv\\test.csv',
                                os.getcwd())

    dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
    print(dataset_sizes, flush=True) # OUTPUT: {'train': 5482, 'test': 1784}


    model = ResTCN().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=50, gamma=.1)

    criterion = nn.CrossEntropyLoss().to(device)
    softmax = nn.Softmax()

    train_losses = []
    val_losses = []

    print("dataloader size ",len(dataloader['train']))
    for epoch in range(num_epochs):
        for phase in ['train', 'test']:

                running_loss = .0
                y_trues = np.empty([0])
                y_preds = np.empty([0])

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for inputs, labels in tqdm(dataloader[phase]):

                    inputs = inputs.to(device)
                    labels = labels.long().squeeze().to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    try:
                        preds = torch.max(softmax(outputs), dim=1)[1]  # Fix here
                        # print('HI')
                    except Exception as e:
                        print (e)
                        continue

                    y_trues = np.append(y_trues, labels.data.cpu().numpy())
                    y_preds = np.append(y_preds, preds.cpu())
                    

                # if phase == 'train':
                #     scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)

                print("[{}] Epoch: {}/{} Loss: {} LR: {}".format(
                    phase, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)
                print('\nconfusion matrix\n' + str(confusion_matrix(y_trues, y_preds)))
                print('\naccuracy\t' + str(accuracy_score(y_trues, y_preds)))


    with open('train_losses_aug_weighted-0.5.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

    with open('val_losses_aug_weighted-0.5.pkl', 'wb') as f:
        pickle.dump(val_losses, f)

    torch.save(model.state_dict(), 'resTCN_model_aug_weighted-0.5.pth')

