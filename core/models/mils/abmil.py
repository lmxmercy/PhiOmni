import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.mils.layers import GlobalAttention, GlobalGatedAttention, create_mlp
from core.models.mils.mil_template import MIL

MODEL_TYPE = 'abmil'


class ABMIL(MIL):
    """
    ABMIL (Attention-based Multiple Instance Learning) model.

    This class implements the core ABMIL architecture, which uses a patch embedding MLP,
    followed by a global attention or gated attention mechanism, and an optional classification head.

    Args:
        in_dim (int): Input feature dimension for each instance (default: 1024).
        embed_dim (int): Embedding dimension after patch embedding (default: 512).
        num_fc_layers (int): Number of fully connected layers in the patch embedding MLP (default: 1).
        dropout (float): Dropout rate applied in the MLP and attention layers (default: 0.25).
        attn_dim (int): Dimension of the attention mechanism (default: 384).
        gate (int): Whether to use gated attention (True) or standard attention (False) (default: True).
        n_classes (int): Number of output classes for the classification head (default: 2).
    """

    def __init__(
            self,
            in_dim: int = 1024,
            embed_dim: int = 512,
            num_fc_layers: int = 1,
            dropout: float = 0.25,
            attn_dim: int = 384,
            gate: int = True,
            n_classes: int = 2,
    ):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, n_classes=n_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_func(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            n_classes=1
        )

        if n_classes > 0:
            self.classifier = nn.Linear(embed_dim, n_classes)
        self.initialize_weights()

    def _check_inputs(self, features):
        if features.dim() == 3 and features.shape[0] > 1:
            raise ValueError(f'current model does not currently support batch size > 1')
        if features.dim() == 2:
            features = features.unsqueeze(0)
        return features

    def forward_attention(
            self,
            h: torch.Tensor,
            attn_mask=None,
            attn_only=True
    ) -> torch.Tensor:
        """
        Compute the attention scores (and optionally the embedded features) for the input instances.

        Args:
            h (torch.Tensor): Input tensor of shape [B, M, D], where B is the batch size,
                M is the number of instances (patches), and D is the input feature dimension.
            attn_mask (torch.Tensor, optional): Optional attention mask of shape [B, M], where 1 indicates
                valid positions and 0 indicates masked positions. If provided, masked positions are set to
                a very large negative value before softmax.
            attn_only (bool, optional): If True, return only the attention scores (A).
                If False, return a tuple (h, A) where h is the embedded features and A is the attention scores.

        Returns:
            torch.Tensor: If attn_only is True, returns the attention scores tensor of shape [B, K, M],
                where K is the number of attention heads (usually 1). If attn_only is False, returns a tuple
                (h, A) where h is the embedded features of shape [B, M, D'] and A is the attention scores.
        """
        h = self.patch_embed(h)
        A = self.global_attn(h)  # B x M x K
        A = torch.transpose(A, -2, -1)  # B x K x M
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min

        if attn_only:
            return A
        return h, A

    def forward_features(
            self,
            h: torch.Tensor,
            attn_mask=None,
            return_attention: bool = True
    ) -> torch.Tensor:
        """
        Compute bag-level features using attention pooling.

        Args:
            h (torch.Tensor): [B, M, D] input features.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            Tuple[torch.Tensor, dict]: Bag features [B, D] and attention weights.
        """
        h, A_base = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)  # A == B x K x M
        A = F.softmax(A_base, dim=-1)  # softmax over N
        h = torch.bmm(A, h).squeeze(dim=1)  # B x K x C --> B x C
        log_dict = {'attention': A_base if return_attention else None}
        return h, log_dict

    def forward_head(
            self,
            h: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: [B x D]-dim torch.Tensor.

        Returns:
            logits: [B x n_classes]-dim torch.Tensor.
        """
        logits = self.classifier(h)
        return logits

    def forward_logits(
            self,
            h: torch.Tensor,
    ) -> torch.Tensor:
        h = self._check_inputs(h)
        wsi_feats, log_dict = self.forward_features(h, attn_mask=None)
        logits = self.forward_head(wsi_feats)
        return logits

    def forward(
            self,
            h: torch.Tensor,
            loss_fn: nn.Module = None,
            label: torch.LongTensor = None,
            attn_mask=None,
            return_attention: bool = False,
            return_slide_feats: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for ABMIL.

        Args:
            h: [B, M, D] input features.
            loss_fn: Optional loss function.
            label: Optional labels.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (results_dict, log_dict) with logits and loss.
        """
        # check the shape of input wsi_feats
        h = self._check_inputs(h)
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)
        results_dict = {'logits': logits, 'loss': cls_loss}
        log_dict['loss'] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict['slide_feats'] = wsi_feats
        return results_dict, log_dict


if __name__ == "__main__":

    path_feat = torch.rand(100, 1536)
    model = ABMIL(in_dim=1536, embed_dim=512, n_classes=4, num_fc_layers=1, dropout=0.2)
    results_dict, log_dict = model(path_feat, label=1, loss_fn=torch.nn.CrossEntropyLoss())
    print(results_dict['logits'].shape)
    print(results_dict['loss'])