from core.models import TANGLE
from core.models import ABMIL, CLAMSB, ILRA, TransMIL
from core.models.phiomni import PhiOmni

def create_mil_model(args):

    model_name = args["model"]
    # model configs taken from https://github.com/mahmoodlab/MIL-Lab/tree/main/src/model_configs
    if model_name.lower() == 'abmil':
        mil_model = ABMIL(in_dim=args["embedding_dim"], gate=True, embed_dim=512, attn_dim=384, num_fc_layers=1,
                          dropout=0.25, n_classes=args["n_classes"])

    elif model_name.lower() == 'clamsb':
        mil_model = CLAMSB(in_dim=args["embedding_dim"], gate=True, embed_dim=512, attention_dim=384, n_fc_layers=1, dropout=0.25,
                           k_sample=8, subtyping=True, instance_loss_fn='ce', bag_weight=0.7, n_classes=args["n_classes"])

    elif model_name.lower() == 'transmil':
        mil_model = TransMIL(in_dim=args["embedding_dim"], embed_dim=512, num_fc_layers=1, dropout=0.25,
                             num_attention_layers=2, num_heads=4, n_classes=args["n_classes"])

    elif model_name.lower() == 'ilra':
        mil_model = ILRA(in_dim=args["embedding_dim"], embed_dim=256, num_attention_layers=2, num_heads=8,
                         topk=64, ln=True, mode='classification', n_classes=args["n_classes"])
    else:
        raise NotImplementedError(f"MIL model {model_name} not implemented.")

    return mil_model


def create_ssl_model(model_name, config,):

    if model_name.lower() in ['tangle', 'tanglerec']:
        ssl_model = TANGLE(config=config)

    elif model_name.lower() == 'phiomni':
        ssl_model = PhiOmni(config=config)
    else:
        raise NotImplementedError(f"SSL model {model_name} not implemented.")

    return ssl_model