"""
Default choices for auxiliary classifications tasks, encoders & decoders.
"""
from referit3d.models import MLP
import torch.nn as nn
from referit3d.models import LSTMEncoder
from referit3d.models import load_glove_pretrained_embedding, make_pretrained_embedding
from referit3d.in_out.vocabulary import Vocabulary
import torchvision.models as models
import argparse
from timm.models import create_model
from referit3d.models.backbone.visual_encoder.pointnet2 import get_model
from referit3d.external_tools.pointnet2.pointnet2 import Pointnet2Backbone
import clip

try:
    from referit3d.models import PointNetPP
except ImportError:
    PointNetPP = None


#
# Object Encoder
#
def image_object_encoder(args: argparse.Namespace, featdim):
    # model = create_model("convnext_tiny", pretrained=True, num_classes=featdim, drop_path_rate=0.1, head_init_scale=1.0)
    # model = nn.Sequential(model, nn.Dropout(0.2))
    model, preprocess = clip.load("ViT-B/16")
    model = model.encode_image
    return model


def pc_object_encoder(args: argparse.Namespace, featdim):
    model = get_model(num_class=args.object_latent_dim, normal_channel=True)  # PyTorch Version
    # model = Pointnet2Backbone(num_class=args.object_latent_dim, input_feature_dim=3)  # Cuda Version 2
    return model


def single_object_encoder(args: argparse.Namespace, featdim):
    """

    @param: out_dims: The dimension of each object feature
    """

    if args.object_encoder == "r50":
        # https://pytorch.org/vision/stable/models.html
        resnet50 = models.resnet50(pretrained=True, num_classes=1000)
        resnet50.fc = nn.Linear(512 * 4, args.object_latent_dim)
        return resnet50
    elif args.object_encoder == "r18":
        # https://pytorch.org/vision/stable/models.html
        resnet18 = models.resnet18(pretrained=True, num_classes=1000)
        resnet18.fc = nn.Linear(512, args.object_latent_dim)
        return resnet18
    elif args.object_encoder == "convnext":
        # https://github.com/facebookresearch/ConvNeXt
        model = create_model("convnext_tiny", pretrained=True, num_classes=featdim, drop_path_rate=0.1,
                             head_init_scale=1.0)
        # model = nn.Sequential(model, nn.Dropout(0.2))

        return model
    elif args.object_encoder == "convnext_p++":
        model1 = create_model("convnext_tiny", pretrained=True, num_classes=featdim,
                              drop_path_rate=0.1, head_init_scale=1.0)
        # model1 = nn.Sequential(model1, nn.Dropout(0.2))
        model2 = get_model(num_class=args.object_latent_dim, normal_channel=True)
        return [model1, model2]
    elif args.object_encoder == "pnet_pp":
        """
        model = PointNetPP(sa_n_points=[32, 16, None], sa_n_samples=[32, 32, None], sa_radii=[0.2, 0.4, None],
                           sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, args.object_latent_dim]])
        """
        model = get_model(num_class=args.object_latent_dim, normal_channel=True)
        return model
    else:
        raise NotImplementedError('Unknown object_encoder model is requested.')

    ## Further improve acc by ~2%. But increase training time & GPU memory
    ## Set default as the lighter version
    # return PointNetPP(sa_n_points=[64, 32, None],
    #                   sa_n_samples=[64, 64, None],
    # return PointNetPP(sa_n_points=[32, 16, None],
    #                   sa_n_samples=[32, 32, None],
    #                   sa_radii=[0.2, 0.4, None],
    #                   sa_mlps=[[3, 64, 64, 128],
    #                            [128, 128, 128, 256],
    #                            [256, 256, 512, out_dim]])


def create_scene_encoder(args: argparse.Namespace):
    # https://github.com/facebookresearch/ConvNeXt
    model = create_model("convnext_tiny", pretrained=True, num_classes=args.object_latent_dim,
                         drop_path_rate=0.1,
                         head_init_scale=1.0, )
    model = nn.Sequential(model, nn.Dropout(0.2))
    return model


#
#  Token Encoder
#
def token_encoder(vocab: Vocabulary,
                  word_embedding_dim: int,
                  lstm_n_hidden: int,
                  word_dropout: float,
                  init_c=None, init_h=None, random_seed=None,
                  feature_type='max',
                  glove_emb_file: str = '') -> LSTMEncoder:
    """
    Language Token Encoder.

    @param vocab: The vocabulary created from the dataset (nr3d or sr3d) language tokens
    @param word_embedding_dim: The dimension of each word token embedding
    @param glove_emb_file: If provided, the glove pretrained embeddings for language word tokens
    @param lstm_n_hidden: The dimension of LSTM hidden state
    @param word_dropout:
    @param init_c:
    @param init_h:
    @param random_seed:
    @param feature_type:
    """
    if len(glove_emb_file) > 0:
        print('Using glove pre-trained embeddings.')
        glove_embedding = load_glove_pretrained_embedding(glove_emb_file, verbose=True)
        word_embedding = make_pretrained_embedding(vocab, glove_embedding, random_seed=random_seed)

        # word-projection here is a bit deeper, since the glove-embedding is frozen.
        word_projection = nn.Sequential(nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU(),
                                        nn.Dropout(word_dropout),
                                        nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU())
    else:
        word_embedding = nn.Embedding(len(vocab), word_embedding_dim, padding_idx=vocab.pad)
        word_projection = nn.Sequential(nn.Dropout(word_dropout),
                                        nn.Linear(word_embedding_dim, word_embedding_dim),
                                        nn.ReLU())

    assert vocab.pad == 0 and vocab.eos == 2

    model = LSTMEncoder(n_input=word_embedding_dim, n_hidden=lstm_n_hidden, word_embedding=word_embedding,
                        init_c=init_c, init_h=init_h, word_transformation=word_projection, eos_symbol=vocab.eos,
                        feature_type=feature_type)
    return model


#
# Object Decoder
#
def object_decoder_for_clf(object_latent_dim: int, n_classes: int) -> MLP:
    """
    The default classification head for the fine-grained object classification.

    @param object_latent_dim: The dimension of each encoded object feature
    @param n_classes: The number of the fine-grained instance classes
    """
    return MLP(object_latent_dim, [128, 256, n_classes], dropout_rate=0.15)


#
#  Text Decoder
#
def text_decoder_for_clf(in_dim: int, n_classes: int) -> MLP:
    """
    Given a text encoder, decode the latent-vector into a set of clf-logits.

    @param in_dim: The dimension of each encoded text feature
    @param n_classes: The number of the fine-grained instance classes
    """
    out_channels = [128, n_classes]
    dropout_rate = [0.2]
    return MLP(in_feat_dims=in_dim, out_channels=out_channels, dropout_rate=dropout_rate)


#
# Referential Classification Decoder Head
#
def object_lang_clf(in_dim: int) -> MLP:
    """
    After the net processes the language and the geometry in the end (head) for each option (object) it
    applies this clf to create a logit.

    @param in_dim: The dimension of the fused object+language feature
    """
    return MLP(in_dim, out_channels=[128, 64, 1], dropout_rate=0.05)
