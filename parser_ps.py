'''
To set arguments
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Set parameters.")

    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed.')
    parser.add_argument('--dataset', type=str, default='patent_6600',
                        help='Specify the dataset. Choose from {patent_2425, patent_6600, patent_10229, patent_17147}')    
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--node_emb_size', type=int, default=64,
                        help='Node embedding size.')
    parser.add_argument('--hidden_emb_size', type=int, default=32,
                        help='LSTM hidden embedding size.')    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--l2_weight', type=float, default=1e-7, 
                        help='weight of l2 regularization')
    parser.add_argument('--use_subgraph', type=bool, default=True,
                        help='Whether to use subgraph or not')
    parser.add_argument('--aggregator', type=str, default='gate',
                        help='How to get the subgraph representation, {avg, ewp, gate}')
    parser.add_argument('--fusion_type', type=str, default='att',
                        help='the way of integrating the subgraph representation with nodes in path, {add, ewp, att}')
    parser.add_argument('--merge_type', type=str, default='att',
                        help='the way of merging the path representations, {avg, att}')
    parser.add_argument('--kge_type', type=str, default='No_KGE',
                        help='choose a KGE method from {TransR, TransE}, choose {No_KGE} if you do not want to use a KGE method')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='batch size for KGE method')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-4,
                        help='weight of l2 regularization used in KGE method')

    parser.add_argument('--evaluate_every', type=int, default=5,
                        help='The epoch interval of model evaluation') 
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='The batch size for test')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    return parser.parse_args()
