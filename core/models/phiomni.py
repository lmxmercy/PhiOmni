import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from einops import rearrange  # type: ignore
from torch import nn  # type: ignore

# --> Internal imports
from core.global_mapping import ENCODER_DIM_MAPPING
from core.models.components.layers import ABMILEmbedder, MLP


class PhiOmni(nn.Module):
    def __init__(self, config):
        super(PhiOmni, self).__init__()

        self.config = config
        self.n_tokens_wsi = config['n_tokens']
        self.patch_embedding_dim = config['embedding_dim']
        self.n_tokens_mol = config["n_tokens_mol"]
        self.n_tokens_rpt = config["n_tokens_rpt"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ---> setup wsi encoder
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
            raise NotImplementedError(f"WSI encoder {self.config['wsi_encoder']} not implemented yet")
        
        # ---> setup transcriptomics encoder
        self.mol_embedder = MLP(
            input_dim=self.n_tokens_mol,
            hidden_dim=self.n_tokens_mol,
            output_dim=self.config["embedding_dim"],
        )

        # ---> setup report encoder
        self.rpt_embedder = MLP(
            input_dim=self.n_tokens_rpt,
            hidden_dim=self.n_tokens_rpt,
            output_dim=self.config["embedding_dim"],
        )

        self.sib = nn.Sequential(
            nn.Linear(self.config["embedding_dim"] * 3, self.config["embedding_dim"] * 2),
            nn.ReLU(),
            nn.Linear(self.config["embedding_dim"] * 2, self.config["embedding_dim"]),
            nn.LayerNorm(self.config["embedding_dim"]),
        )

    def get_wsi_feature(self, wsi_emb):
        wsi_emb = self.wsi_embedder(wsi_emb)
        return wsi_emb

    def get_slide_attention(self, wsi_emb):
        _, attention = self.wsi_embedder(wsi_emb, return_attention=True)
        return attention

    def forward(self, wsi_emb, mol_emb=None, rpt_emb=None):
        wsi_emb = self.wsi_embedder(wsi_emb)
        mol_emb = self.mol_embedder(mol_emb) if mol_emb is not None else torch.zeros_like(wsi_emb)
        rpt_emb = self.rpt_embedder(rpt_emb) if rpt_emb is not None else torch.zeros_like(wsi_emb)

        mm_emb = torch.cat([wsi_emb, mol_emb, rpt_emb], dim=1)
        mm_emb = self.sib(mm_emb)

        return wsi_emb, mol_emb, rpt_emb, mm_emb