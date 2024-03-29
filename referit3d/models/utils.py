import torch
import torch.nn.functional as F


def get_siamese_features(net, in_features, aggregator=None, mode=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        if mode is not None:
            out_features.append(net(in_features[:, i], mode=mode))
        else:
            out_features.append(net(in_features[:, i]))
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """Save torch items with a state_dict.
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, strict=True, args=None, **kwargs):
    """Load torch items from saved state_dictionaries.
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key], strict=strict)
    if args.twoStreams:
        clean_dict = {key.replace("module.", ''): item for key, item in checkpoint[key].items()}
        clean_dict = {key.replace("object_encoder.", ''): item for key, item in clean_dict.items()}
        if args.multiprocessing_distributed and False:
            value.module.img_object_encoder.load_state_dict(clean_dict, strict=strict)
        else:
            value.img_object_encoder.load_state_dict(clean_dict, strict=strict)

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch
