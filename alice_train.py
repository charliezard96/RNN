# -*- coding: utf-8 -*-

import argparse
import torch
import json
from torch import nn
from alice_dataset import AliceDataset, RandomCrop, OneHotEncoder, ToTensor
from alice_network import Network, train_batch, validation_batch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from pathlib import Path
import numpy as np


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the chapter generator network.')

# Dataset
parser.add_argument('--datasetpath',    type=str,   default='alice.txt', help='Path of the train txt file')
parser.add_argument('--crop_len',       type=int,   default=100,               help='Number of input letters')
parser.add_argument('--alphabet_len',   type=int,   default=35,                help='Number of letters in the alphabet')

# Network
parser.add_argument('--hidden_units',   type=int,   default=256,    help='Number of RNN hidden units')
parser.add_argument('--layers_num',     type=int,   default=2,      help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',   type=float, default=0.3,    help='Dropout probability')

# Training
parser.add_argument('--batchsize',      type=int,   default=1000,   help='Training batch size')
parser.add_argument('--num_epochs',     type=int,   default=20000,    help='Number of training epochs')

# Save
parser.add_argument('--out_dir',     type=str,   default='alice_save',    help='Where to save models and params')

##############################
##############################
##############################

 #%%
    # creation of the k-fold (k=3)
def create_kFold(dataset):
        
    len_data = int(len(dataset))
    i_list = list(np.arange(0,len(dataset)))
    
    vali1 = DataLoader(Subset(dataset, i_list[0:int(len_data/3)]), 
                       batch_size=args.batchsize, shuffle=True);

    vali2 = DataLoader(Subset(dataset, i_list[int(len_data/3): int(len_data/3)*2]), 
                       batch_size=args.batchsize, shuffle=True);

    vali3 = DataLoader(Subset(dataset, i_list[int(len_data/3)*2:len_data]), 
                       batch_size=args.batchsize, shuffle=True);
    
    tran1 = DataLoader(Subset(dataset, i_list[0:int(len_data/3)]), 
                       batch_size=args.batchsize, shuffle=True)
    
    tran3 = DataLoader(Subset(dataset, i_list[0:int(len_data/3)*2]), 
                       batch_size=args.batchsize, shuffle=True)
    
    i_list_for2 = i_list[0:int(len_data/3)] + i_list[int(len_data/3)*2:len_data]
    tran2 = DataLoader(Subset(dataset, i_list_for2), 
                       batch_size=args.batchsize, shuffle=True)
    kFold_dataset = [
            [tran1, vali1],  
            [tran2, vali2], 
            [tran3,vali3]
            ]
    
    return kFold_dataset
#%%
def swap(val1, val2):
    valf1 = val2
    valf2 = val1
    return valf1,valf2    
#%%
if __name__ == '__main__':
    
    # Parse input arguments
    args = parser.parse_args()
    
    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    #%% Create dataset
    trans = transforms.Compose([RandomCrop(args.crop_len),
                                OneHotEncoder(args.alphabet_len),
                                ToTensor()
                                ])
    
    dataset = AliceDataset(filepath=args.datasetpath, crop_len=args.crop_len,transform=trans)
    
   
    #%% set up phase
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # hyp param are hidden units, layer number, learning rate
    chosen_hyp_params = []
    opt_loss = 1000
    attempt = 1
    best_net = -1
    loss_kfold_th = []
    num_epochs_for_kfold = 5000
    #%% K-FOLD NOT SYSTEMATIC SEARCH (param range is arbitrary)
    
    crop_len = args.crop_len    
    
    trans = transforms.Compose([RandomCrop(crop_len),
                    OneHotEncoder(args.alphabet_len),
                    ToTensor()
                    ])
        
    for hidden_units in [64,128,256]:
            for layer_number in [2,3,5]:
                for lrate in [0.1, 0.01, 0.001]:
                    print('\n Net: ' + str(attempt))
                    net = Network(input_size=args.alphabet_len, 
                                  hidden_units=hidden_units, 
                                  layers_num=layer_number, 
                                  dropout_prob=args.dropout_prob)
                    net.to(device)
                    optimizer = torch.optim.RMSprop(net.parameters(), lr = lrate)      
                    kfold = create_kFold(dataset);
                    low_loss = True
                    loss_rec = []
                    
                    for train_loader, val_loader in kfold:
                        if(low_loss):
                            # Start training
                            print("Current net params: " + str(hidden_units) + " " +str(layer_number) + " "  +str(lrate))
                            batch_loss_max = 4;
                            
                            for epoch in range(num_epochs_for_kfold):
                                running_loss_val = 0;
                                if (epoch + 1) % 2500 == 0:
                                    print('## EPOCH %d' % (epoch + 1))
                                          
                                # Iterate batches
                                
                                for batch_sample in train_loader:
                                    # Extract batch
                                    batch_onehot = batch_sample['encoded_onehot'].to(device);
                                    # Update network
                                    batch_loss =  train_batch(net, batch_onehot, loss_fn, optimizer);
                                
                                if(batch_loss > batch_loss_max):
                                    loss_rec.append(1000)
                                    low_loss = False
                                    break
                                    
                                if (epoch + 1) % 2500 == 0: #elimina
                                    print('\t Training loss (single batch):', batch_loss)
                                    
                                for val_batch_sample in val_loader:
                                     # Extract batch
                                    val_batch_sample = val_batch_sample['encoded_onehot'].to(device);
    
                                    val_loss =  validation_batch(net, batch_onehot, loss_fn, optimizer);
                                    running_loss_val = running_loss_val + val_loss
                                    if (epoch == num_epochs_for_kfold -1):
                                        loss_rec.append(running_loss_val/len(val_loader))
                                        print('\t Avg Validation loss, 3-fold:', running_loss_val/len(val_loader))
                        else:
                            loss_rec.append(1000)
                    loss_valid_avg = np.mean(np.array(loss_rec))
                    print("mean loss of the current Net: " + str(loss_valid_avg))
                    loss_kfold_th.append(loss_valid_avg)
                    
                    if (loss_valid_avg < opt_loss):
                        opt_loss = loss_valid_avg
                        best_net = attempt
                        chosen_hyp_params = [hidden_units,layer_number,lrate]  
                        
                    attempt = attempt + 1
                    
    print("The best Net is #" + str(best_net))
    print(chosen_hyp_params)
    print(min(np.asarray(loss_kfold_th)))
    #%% training of network
    
    chosen_hyp_params2 = chosen_hyp_params 
    trans = transforms.Compose([RandomCrop(crop_len=args.crop_len),
                                OneHotEncoder(args.alphabet_len),
                                ToTensor()
                                ])
    net = Network(input_size=args.alphabet_len, 
                  hidden_units=chosen_hyp_params2[0], 
                  layers_num=chosen_hyp_params2[1], 
                  dropout_prob=args.dropout_prob)
    net.to(device)
    # Define Dataloader
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    # Define optimizer
    optimizer = torch.optim.RMSprop(net.parameters(), lr = chosen_hyp_params2[2])
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    #improving speed of training
    is_improved = True
    last_improve = 1000
    count_no_improve = 0
    improve_tresh = 1000
    
    # Start training
    for epoch in range(args.num_epochs):
        if (epoch + 1) % 500 == 0:
            print('##################################')
            print('## EPOCH %d' % (epoch + 1))
            print('##################################')
        if(epoch > 10000):
            improve_tresh = 50
        if(epoch > 13000):
            improve_tresh = 10    
        if(is_improved):
            
            # Iterate batches
            for batch_sample in dataloader:
                # Extract batch
                batch_onehot = batch_sample['encoded_onehot'].to(device)
                # Update network
                batch_loss = train_batch(net, batch_onehot, loss_fn, optimizer)
#                print('\t Training loss (single batch):', batch_loss)
                if(last_improve < batch_loss):
                    count_no_improve = count_no_improve + 1
#                    print('\t count_no_improve:', count_no_improve)
                    if(count_no_improve >= improve_tresh):
                        is_improved = False
                        break
                else:
                    count_no_improve = 0
                    last_improve = batch_loss 
                                
                if (epoch + 1) % 500 == 0:
                        print('\t Training loss (single batch):', batch_loss)
        
        else:
            break
    
    #%%
    ### Save all needed parameters
    # Create output dir
    out_dir = Path("new_alice_model_" + str(epoch +1));
    out_dir.mkdir(parents=True, exist_ok=True);
    # Save network parameters
    torch.save(net.state_dict(), out_dir / 'net_params.pth');
    # Save training parameters
    with open(out_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    # Save encoder dictionary
    with open(out_dir / 'char_to_number.json', 'w') as f:
        json.dump(dataset.char_to_number, f, indent=4)
    # Save decoder dictionary
    with open(out_dir / 'number_to_char.json', 'w') as f:
        json.dump(dataset.number_to_char, f, indent=4)
    
    
    

        
