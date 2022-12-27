import argparse

def init_args():

    parser = argparse.ArgumentParser(description='Graph Latent Augmentation for Contrastive Learning')

    parser.add_argument('--model', metavar='M', type=str,
        choices=['GAT', 'GCN'], default='GCN', help='The model to use for embedding.')
    
    # Experiment Settings
    parser.add_argument('--dataset', type=str, default='Computers', help='dataset to be used:\
                         Cora Citeseer PubMed | Computers Photo') 
    parser.add_argument('--repeat', type=int, default=10, help='Repeat the experiment for n times.')
    parser.add_argument('--seed', type=int, default=0, help='The manual seed for experiments.')
    parser.add_argument('--best_k_epochs', type=int, default=1, help='Repeat the experiment for n times.')
    parser.add_argument('--tune', type=bool, default=False, help='if we are under tuning mode.') 
    
    # Model Design
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers.')
    parser.add_argument('--hidden_size', type=int, default=256, help='The hidden size.')
    parser.add_argument('--emb_size', type=int, default=256,  help='The embedding size.')

    parser.add_argument('--base_model', type=str, default='GATConv',  help='base model type')
    parser.add_argument('--activation', type=str, default='LeakyReLU',  help='activation type')
    parser.add_argument('--enable_encoder_skip', type=bool, default=False,  help='enable encoder skip')


    
    parser.add_argument('--dropout', type=float, default=0.5, help='The dropout ratio for Encoder.')
    parser.add_argument('--vgae_dropout', type=float, default=0.0, help='The dropout ratio for VGAE.')
    parser.add_argument('--reconstruct', type=str, default='A', help='Reconstruct the Adj (A) or Feature (X) in VGAE')
    parser.add_argument('--proj_head', type=str, default='FF', help='use FF or Linear Projection head')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--classifier_epochs', type=int, default=100)
    parser.add_argument('--normalize', type=bool, default=False, help='if to normalize the embedding.')
    parser.add_argument('--num_samples', type=int, default=200, help='number of samples for contrastive loss calculation (limited by GPU memory)') 
    parser.add_argument('--wa', type=int, default=5, help='top k val acc for weight averaging') 
    
    parser.add_argument('--lr', type=float, default=0.05, help='The learning rate.')
    parser.add_argument('--classifier_lr', type=float, default=0.05, help='The learning rate.')
    parser.add_argument('--vgae_lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--lr_step_rate', type=float, default=1, help='The learning rate step rate (gamma).')
    parser.add_argument('--l2', type=float, default=5e-3, help='The l2 regularizer.')
    parser.add_argument('--vgae_l2', type=float, default=0, help='The l2 regularizer for vgae (dont set too large).')
    parser.add_argument('--classifier_l2', type=float, default=5e-4, help='The l2 regularizer for Classifier.')
    parser.add_argument('--flood', type=float, default=0, help='Threshold of flooding loss.')
    
    # Model Hyperparameters
    parser.add_argument('--tau', type=float, default=1.0,help='temperature in the contrastive loss.')
    parser.add_argument('--gamma', type=float, default=1, help='gamma for the contrastive loss.')
    
    return parser.parse_args()
