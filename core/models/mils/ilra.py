import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.mils.mil_template import MIL

"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""

class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K, return_attention=False):
        """
        Args:
            Q: (B, S_Q, D_Q)
            K: (B, S_K, D_K)
        Returns:
            O: (B, S_Q, D_V) - output after attention
            A: (B, S_Q, S_K) - attention scores
        """

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)
        A, attention_weights = self.multihead_attention(Q, K, V,
                                                        need_weights=return_attention,
                                                        average_attn_weights=True)  # A is shaped S_Q, B, D_V
        attention_weights = attention_weights.transpose(0, 1) if attention_weights is not None else None

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O, attention_weights


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H, _ = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat, _ = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat


class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method.md
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, return_attention=False):
        global_embedding = self.S.repeat(X.size(0), 1, 1)  # expand to batch dim
        ret, attention = self.mha(global_embedding, X, return_attention=return_attention)  # cross attention scores
        if return_attention:
            attention = torch.sum(attention, dim=1)  # B x patches
        return ret, attention


class ILRA(MIL):
    def __init__(self, in_dim, embed_dim, num_heads,
                 topk, num_attention_layers, n_classes, ln=True, mode='classification'):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, n_classes=n_classes)
        self.mode = mode
        topk = topk

        self.mlp = None

        gab_blocks = []
        for idx in range(num_attention_layers):
            block = GAB(dim_in=in_dim if idx == 0 else embed_dim,
                        dim_out=embed_dim,
                        num_heads=num_heads,
                        num_inds=topk,
                        ln=ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=embed_dim, num_heads=num_heads, ln=ln)

        # classifier
        self.classifier = nn.Linear(in_features=embed_dim, out_features=n_classes)

        self.initialize_weights()

    def reset_classifier(self):
        self.classifier.reset_parameters()

    def forward_features(self, x, return_attention=False):
        for block in self.gab_blocks:
            x = block(x)
        slide_feat, attention = self.forward_attention(x, return_attention=return_attention)
        return slide_feat, {'attention': attention}

    def forward_attention(self, x, return_attention):
        slide_feat, attention = self.pooling(x, return_attention)
        return slide_feat, attention

    def forward_head(self, slide_feats):
        logits = self.classifier(slide_feats)  # [B x n_classes]
        logits = logits.squeeze(1)
        return logits

    def forward_logits(
            self,
            h: torch.Tensor,
    ) -> torch.Tensor:
        slide_feats, _ = self.forward_features(h, return_attention=False)
        logits = self.forward_head(slide_feats)
        return logits

    def forward(self, h, label: torch.LongTensor=None,
                loss_fn: nn.Module=None,
                return_attention=False,
                **kwargs):
        if self.mode == 'classification':
            slide_feats, attention = self.forward_features(h, return_attention=return_attention)
            logits = self.forward_head(slide_feats)
            cls_loss = MIL.compute_loss(loss_fn, logits, label)
            results_dict = {'logits': logits, 'loss': cls_loss}
            log_dict = {'loss': cls_loss.item() if cls_loss is not None else -1, }
            if return_attention:
                log_dict['attention'] = attention

        elif self.mode == 'survival': # todo
            attention_mask = kwargs['attn_mask']
            label = kwargs['label']
            censorship = kwargs['censorship']
            loss_fn = kwargs['loss_fn']

            out = self.forward_no_loss(h)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict