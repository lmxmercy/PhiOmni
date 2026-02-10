# --> Torch imports
import torch
from torch import nn

# --> Internal imports
from core.models.components.layers import MLP, ABMILEmbedder


class TANGLE(nn.Module):
    def __init__(self, config):
        super(TANGLE, self).__init__()

        self.config = config
        self.n_tokens_wsi = config['n_tokens'] 
        self.patch_embedding_dim = config['embedding_dim']
        self.n_tokens_rna = config['rna_token_dim']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ########## WSI embedder #############
        if self.config["wsi_encoder"] == "abmil":
            assert self.config["n_heads"] == 1, "ABMIL must have only 1 head"
            pre_params = {
                "input_dim": self.patch_embedding_dim,
                "hidden_dim": self.patch_embedding_dim,
            }
            attention_params = {
                "model": "ABMIL",
                "params": {
                    "input_dim": self.patch_embedding_dim,
                    "hidden_dim": self.config["hidden_dim"],
                    "dropout": True,
                    "activation": self.config["activation"],
                    "n_classes": 1,
                },
            }
            self.wsi_embedder = ABMILEmbedder(
                pre_attention_params=pre_params,
                attention_params=attention_params,
            )
        else:
            raise NotImplementedError("WSI encoder {} not implemented".format(self.config["wsi_encoder"]))

        ########## RNA embedder: Linear or MLP #############
        if self.config["rna_encoder"] == "mlp":
            if self.config["study"] == "tanglev2":
                self.rna_embedder = MLP(input_dim=self.n_tokens_rna, hidden_dim=self.config["hidden_dim"], output_dim=self.config["hidden_dim"])
            else:
                self.rna_embedder = MLP(input_dim=self.n_tokens_rna, hidden_dim=self.n_tokens_rna, output_dim=self.config["embedding_dim"])
                # self.rna_embedder = MLP(input_dim=self.n_tokens_rna, hidden_dim=self.config["hidden_dim"], output_dim=self.config["hidden_dim"])

        ########## RNA Reconstruction module: Linear or MLP #############
        if self.config["rna_reconstruction"]:
            if self.config["rna_encoder"] == "linear":
                self.rna_reconstruction = nn.Linear(in_features=self.config["embedding_dim"], out_features=self.n_tokens_rna)
            else:
                self.rna_reconstruction = MLP(input_dim=self.config["embedding_dim"], hidden_dim=self.config["embedding_dim"], output_dim=self.n_tokens_rna)
        else:
            self.rna_reconstruction = None
        
    def forward(self, wsi_emb, rna_emb=None):
        # wsi_emb: [batch_size, n_tokens, n_patches]
        wsi_emb = self.wsi_embedder(wsi_emb) # [batch_size, n_patches]: 128, 1536
        # if self.mean_projector:
        #     wsi_emb = self.mean_projector(wsi_emb)

        # rna_emb: [batch_size, n_tokens_rna]
        if self.config["intra_modality_wsi"] or rna_emb is None or self.config['rna_reconstruction']:
            rna_emb = None
        else:
            rna_emb = self.rna_embedder(rna_emb) # [batch_size, n_patches]: 128, 1536
        
        if self.config["rna_reconstruction"]:
            rna_reconstruction = self.rna_reconstruction(wsi_emb)
        else:
            rna_reconstruction = None

        return wsi_emb, rna_emb, rna_reconstruction
    
    def get_features(self, wsi_emb):
        wsi_emb = self.wsi_embedder(wsi_emb)
        return wsi_emb
        
    def get_slide_attention(self, wsi_emb):
        _, attention = self.wsi_embedder(wsi_emb, return_attention=True)
        return attention
        
    def get_expression_features(self, rna_emb):
        rna_emb = self.rna_embedder(rna_emb)
        return rna_emb
