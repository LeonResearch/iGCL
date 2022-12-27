import argparse

def init_args():

    parser = argparse.ArgumentParser(description='Graph Contrastive Learning with Implicit Augmentations')
    
    # Experiment Settings
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset to be used') 
    parser.add_argument('--repeat', type=int, default=0, help='Repeat the experiment for n times.')
    parser.add_argument('--seed', type=int, default=3, help='The manual seed for experiments.')
    
    # Model Design
    parser.add_argument('--hidden_size', type=int, default=256, help='The hidden size.')
    parser.add_argument('--emb_size', type=int, default=256,  help='The embedding size.')
    parser.add_argument('--dropout', type=float, default=0.5, help='The dropout ratio for Encoder.')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--normalize', type=bool, default=False, help='if to normalize the embedding.')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples for contrastive loss calculation (limited by GPU memory)') 
    parser.add_argument('--wa', type=int, default=5, help='top k val acc for weight averaging') 
    
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate.')
    parser.add_argument('--lr_step_rate', type=float, default=1, help='The learning rate step rate (gamma).')
    parser.add_argument('--l2', type=float, default=5e-3, help='The l2 regularizer.')
    parser.add_argument('--flood', type=float, default=0, help='Threshold of flooding loss.')
    
    # Model Hyperparameters
    parser.add_argument('--tau', type=float, default=1,help='temperature in the contrastive loss.')
    
    return parser.parse_args()
