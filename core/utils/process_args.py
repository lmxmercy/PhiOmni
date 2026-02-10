import argparse


def process_args():

    parser = argparse.ArgumentParser(description='Configurations for Omni-Modal Slide Representation Learning Pretraining')

    # ---> generic
    parser.add_argument('--proj_dir', type=str, default=r"D:\Research Projects\PhiOmni", help='Project directory')
    parser.add_argument('--feats_dir', type=str, default=r"F:\UNI2-h-features")
    parser.add_argument('--study', type=str, default='brca', help='Study: brca, nsclc or pancancer',
                        choices=['pancancer', 'brca', 'nsclc', ])
    parser.add_argument('--type_of_rna', type=str, default='combine', choices=['xena', 'hallmarks', 'combine'])
    parser.add_argument('--type_of_report', type=str, default='gatortron', choices=['gatortron', 'qwen'])

    #-----> model args
    parser.add_argument('--model', type=str, default='tangle', help='SSL or MIL model.',
                        choices=['tangle', 'tanglerec', # SSLs
                                 'abmil', 'clamsb', 'transmil', 'ilra'] # MILs
                        )
    parser.add_argument('--embedding_dim', type=int, default=1536, help='Size of the embedding space')
    parser.add_argument('--rna_encoder', type=str, default="mlp", help='MLP or Linear.')
    parser.add_argument('--sampling_strategy', type=str, default="random", help='How to draw patch embeddings.')
    parser.add_argument('--wsi_encoder', type=str, default="abmil", help='Type of MIL.')
    parser.add_argument('--n_heads', type=int, default=1, help='Number of heads in ABMIL.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Internal dim of ABMIL.')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function used in ABMIL attention weight agg (sigmoid or softmax).')
    parser.add_argument('--mask_percentage', type=float, default=0.5, help='Percentage of n_tokens that is masked during Intra loss computation.')

    #----> training args
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Tensor dtype. Defaults to bfloat16 for increased batch size.')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=2025, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--gpu_devices', type=list, default=[0], help='List of GPUs.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--n_tokens', type=int, default=2048, help='Number of patches to sample during training.')
    parser.add_argument('--symmetric_cl', type=bool, default=True, help='If use symmetric contrastive objective.')
    parser.add_argument('--num_workers', type=int, default=20, help='number of cpu workers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--feature_type', type=str, default='uni_feats', help='What type of features are you using?')

    # ----> log args
    parser.add_argument('--log_wandb', action='store_true', default=False, help='choose if we want to log results in wandb')

    # TANGLE-specific
    parser.add_argument('--intra_modality_mode_wsi', type=str, default='reconstruct_masked_emb',
                        help='Type of Intra loss. Options are: reconstruct_avg_emb, reconstruct_masked_emb.')

    args = parser.parse_args()

    return args