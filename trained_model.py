import json
import torch
from torch import nn
import argparse
from alice_network import Network
from alice_dataset import encode_text, create_one_hot_matrix
from pathlib import Path
import numpy as np

##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate a chapter starting from a given text')

parser.add_argument('--seed', type=str, default='The White Rabbit', help='Initial text of the chapter')

parser.add_argument('--length', type=int, default=100, help='Length')
##############################
##############################
##############################

def sampling_softmax(input_tensor,temperature):
    eps = 1e-16
    q_i = (input_tensor+eps)/temperature
    soft_res = nn.functional.softmax(q_i, dim=1)
    soft_res_float = torch.as_tensor(soft_res).float()
    predict = torch.multinomial(soft_res_float, 1)
    return predict.item()

if __name__ == '__main__':
    
    ### Parse input arguments
    args = parser.parse_args()
    #%% Load training parameters
    model_dir = Path('new_alice_model_10003')
    print ('Loading model from: %s' % model_dir)
    training_args = json.load(open(model_dir / 'training_args.json'))
    #%% Load encoder and decoder dictionaries
    number_to_char = json.load(open(model_dir / 'number_to_char.json'))
    char_to_number = json.load(open(model_dir / 'char_to_number.json'))
    #%% Initialize network
    net = Network(input_size=training_args['alphabet_len'], 
                  hidden_units=training_args['hidden_units'], 
                  layers_num=training_args['layers_num'])
    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))
    net.eval() # Evaluation mode (e.g. disable dropout)    
    #%% Define manually temperature
    temperature = 0.7
    #%% Find initial state of the RNN
    with torch.no_grad():
        # Encode seed
        seed_encoded = encode_text(char_to_number, args.seed)
        # One hot matrix
        seed_onehot = create_one_hot_matrix(seed_encoded, training_args['alphabet_len'])
        # To tensor
        seed_onehot = torch.tensor(seed_onehot).float()
        # Add batch axis
        seed_onehot = seed_onehot.unsqueeze(0)
        # Forward pass
        net_out, net_state = net(seed_onehot)
        # Sample from softmax last output index
        next_char_encoded = sampling_softmax(net_out[:, -1, :], temperature)
        # Print the seed letters
        print(args.seed, end='', flush=True)
        next_char = number_to_char[str(next_char_encoded)]
        print(next_char, end='', flush=True)
        
    #%% Generate text
    new_line_count = 0
    tot_char_count = 0
    while True:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the one hot encoding of the last chosen letter
            net_input = create_one_hot_matrix([next_char_encoded], training_args['alphabet_len'])
            net_input = torch.tensor(net_input).float()
            net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(net_input, net_state)
            # Sample from softmax last output index
            next_char_encoded = sampling_softmax(net_out[:, -1, :], temperature)
            # Decode the letter
            next_char = number_to_char[str(next_char_encoded)]
            print(next_char, end='', flush=True)
            # Count total letters
            tot_char_count += 1
            # Break if n letters
            if tot_char_count > int(args.length):
                break
    