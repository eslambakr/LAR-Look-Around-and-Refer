import torch
import random
import argparse
from torch import nn
from collections import defaultdict

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)
from .mmt_module import *

from . import DGCNN
from .default_blocks import *
from .utils import get_siamese_features
from ..in_out.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


class MMT_ReferIt3DNet(nn.Module):
    def __init__(self,
                 args,
                 visudim=128,
                 MMT_HIDDEN_SIZE=192,
                 TEXT_BERT_HIDDEN_SIZE=768,
                 object_language_clf=None,
                 language_clf=None,
                 object_clf=None,
                 object_clf_2D=None,
                 context_2d=None,
                 feat2dtype=None,
                 mmt_mask=None,
                 n_obj_classes=None):

        super().__init__()
        self.args = args
        self.args_mode = args.mode
        self.text_length = args.max_seq_len
        self.context_2d = context_2d
        self.feat2dtype = feat2dtype
        self.mmt_mask = mmt_mask
        self.train_vis_enc_only = args.train_vis_enc_only
        self.fuse_conv1D = False

        # Encoders for visual 2D objects
        ROIFeatDim = args.feat2ddim
        self.num_class_dim = 525 if '00' in args.scannet_file else 608
        featdim = 0
        if 'ROI' in (args.feat2d.replace('3D', '')):
            featdim += ROIFeatDim
        if 'clsvec' in (args.feat2d.replace('3D', '')):
            featdim += self.num_class_dim
        if 'Geo' in (args.feat2d.replace('3D', '')):
            featdim += 30

        if args.twoStreams:
            self.img_object_encoder = image_object_encoder(args, featdim=ROIFeatDim)
            self.pc_object_encoder = pc_object_encoder(args, featdim=ROIFeatDim)
        elif args.img_encoder:
            self.img_object_encoder = image_object_encoder(args, featdim=ROIFeatDim)
        else:
            self.pc_object_encoder = pc_object_encoder(args, featdim=ROIFeatDim)

        if args.sceneCocoonPath:
            self.img_scene_encoder = image_object_encoder(args, featdim=MMT_HIDDEN_SIZE)
            # self.sceneCocoonFuse = nn.Sequential(nn.Conv1d(MMT_HIDDEN_SIZE, MMT_HIDDEN_SIZE, 5), nn.ReLU())

        if args.train_vis_enc_only == False:
            self.linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
            self.linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)
            self.obj_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
            self.obj_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
            self.obj_drop = nn.Dropout(0.1)
            if args.geo3d:
                self.linear_obj_3dgeofeat_to_mmt_in = nn.Linear(MMT_HIDDEN_SIZE+24, MMT_HIDDEN_SIZE)
                self.obj_3dgeofeat_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)

            if self.args.clspred3d:
                print("3D Cls_Pred is activated !!!!")
                self.linear_3d_feat_to_mmt_in_clspred = nn.Linear(MMT_HIDDEN_SIZE+self.num_class_dim-1, MMT_HIDDEN_SIZE)
                self.obj3d_feat_layer_norm_clspred = BertLayerNorm(MMT_HIDDEN_SIZE)

            # Encoders for text
            self.text_bert_config = BertConfig(
                hidden_size=TEXT_BERT_HIDDEN_SIZE,
                num_hidden_layers=3,
                num_attention_heads=12,
                type_vocab_size=2)
            self.text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=self.text_bert_config, mmt_mask=self.mmt_mask)
            if TEXT_BERT_HIDDEN_SIZE != MMT_HIDDEN_SIZE:
                self.text_bert_out_linear = nn.Linear(TEXT_BERT_HIDDEN_SIZE, MMT_HIDDEN_SIZE)
            else:
                self.text_bert_out_linear = nn.Identity()

        if self.context_2d == 'unaligned':
            self.linear_2d_feat_to_mmt_in = nn.Linear(featdim, MMT_HIDDEN_SIZE)
            self.obj2d_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
            if 'clspred' in (args.feat2d.replace('3D', '')):
                print("Cls_Pred is activated !!!!")
                self.linear_2d_feat_to_mmt_in_clspred = nn.Linear(MMT_HIDDEN_SIZE+self.num_class_dim-1, MMT_HIDDEN_SIZE)
                self.obj2d_feat_layer_norm_clspred = BertLayerNorm(MMT_HIDDEN_SIZE)
            # self.linear_2d_bbox_to_mmt_in = nn.Linear(30, MMT_HIDDEN_SIZE)
            # self.obj2d_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
            # self.linear_2d_bbox_to_mmt_in = nn.Linear(30, 256)
            # self.obj2d_bbox_layer_norm = BertLayerNorm(256)
            self.context_drop = nn.Dropout(0.1)

        # classifier heads
        if args.train_vis_enc_only == False:
            self.object_clf = object_clf
            self.language_clf = language_clf
            self.object_language_clf = object_language_clf
            outputdim = 1
            self.matching_cls = MatchingLinear(input_size=MMT_HIDDEN_SIZE, outputdim=outputdim)  # Final 3D Ref
        self.object_clf_2D = object_clf_2D
        self.img_encoder = args.img_encoder
        self.args = args

        if args.cocoon:
            # TODO: Eslam should make this generic
            self.cocoon_fusion = nn.Conv1d(ROIFeatDim, ROIFeatDim, 5)

        if args.train_vis_enc_only == False:
            self.mmt_config = BertConfig(
                hidden_size=MMT_HIDDEN_SIZE,
                num_hidden_layers=4,
                num_attention_heads=12,
                type_vocab_size=2)
            if args.twoTrans:
                if self.fuse_conv1D:
                    self.ref_fusion3D_2D = nn.Conv1d(MMT_HIDDEN_SIZE, MMT_HIDDEN_SIZE, 2)
                else:
                    if args.sceneCocoonPath:
                        self.ref_fusion3D_2D = MMT_3D_2D_scene(self.mmt_config, context_2d=self.context_2d,
                                                               mmt_mask=self.mmt_mask)
                    else:
                        self.ref_fusion3D_2D = MMT_vision(self.mmt_config, context_2d=self.context_2d,
                                                          mmt_mask=self.mmt_mask)

                self.matching_cls_3D_beforeFusion = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
                self.mmt_vis = MMT_vision(self.mmt_config, context_2d=self.context_2d, mmt_mask=self.mmt_mask)
                self.mmt_vis_lang = MMT_VisLang(self.mmt_config, context_2d=self.context_2d, mmt_mask=self.mmt_mask)
                if self.args.sharetwoTrans==False:
                    self.mmt_vis_lang_2D = MMT_VisLang(self.mmt_config, context_2d=self.context_2d,
                                                       mmt_mask=self.mmt_mask)
            else:
                self.mmt = MMT(self.mmt_config, context_2d=self.context_2d, mmt_mask=self.mmt_mask)

            if self.context_2d == 'unaligned':
                outputdim = 1
                self.matching_cls_2D = MatchingLinear(input_size=MMT_HIDDEN_SIZE, outputdim=outputdim)  # Final 2D ref
                if args.twoTrans:
                    self.matching_cls_2D_beforeFusion = MatchingLinear(input_size=MMT_HIDDEN_SIZE)

    def dummy_lambda(self):
        return None

    def __call__(self, batch: dict, mode) -> dict:
        # result = defaultdict(lambda: None)
        result = defaultdict(self.dummy_lambda)
        # 2D encoder:
        if self.img_encoder:
            if self.args.cocoon:
                # [b, N-objs, cocoon_imgs, h, w, c] --> [b, N-objs, cocoon_imgs, c, h, w]
                batch['objectsImgs'] = batch['objectsImgs'].permute(0, 1, 2, 5, 3, 4).float()
                objects_features = []
                for i in range(batch['objectsImgs'].shape[1]):  # loop on N-objs
                    objImgs = batch['objectsImgs'][:, i]  # [b, cocoon_imgs, c, h, w]
                    out_features = get_siamese_features(self.img_object_encoder, objImgs,
                                                        aggregator=torch.stack)  # [b, cocoon_imgs, embed_dim]
                    out_features = out_features.permute(0, 2,
                                                        1)  # [b, cocoon_imgs, embed_dim]-->[b, embed_dim, cocoon_imgs]
                    out_features = self.cocoon_fusion(out_features).squeeze()
                    # out_features = out_features.sum(dim=1)  # [b, embed_dim]
                    objects_features.append(out_features)
                # stack all object features: [b, embed_dim] --> [b, N-objs, embed_dim]
                objects_img_features = torch.stack(objects_features, dim=1)
            else:
                # Single image:
                # [b, N-objs, h, w, c] --> [b, N-objs, c, h, w]
                batch['objectsImgs'] = batch['objectsImgs'].permute(0, 1, 4, 2, 3).float()
                objects_img_features = get_siamese_features(self.img_object_encoder, batch['objectsImgs'],
                                                            aggregator=torch.stack)  # [b, N-objs, embed_dim]
        # 3D encoder:
        if self.args.twoStreams or not self.img_encoder:
            # Get features for each segmented scan object based on color and point-cloud
            objects_pc_features = get_siamese_features(self.pc_object_encoder, batch['objects'], mode=mode,
                                                       aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Whole scene encoder:
        if self.args.sceneCocoonPath:
            # [b, cocoon_imgs, h, w, c] --> [b, cocoon_imgs, c, h, w]
            batch['sceneImgs'] = batch['sceneImgs'].permute(0, 1, 4, 2, 3).float().contiguous()
            # [b, cocoon_imgs, c, h, w] --> [b, cocoon_imgs, embed_dim]
            scene_features = get_siamese_features(self.img_scene_encoder, batch['sceneImgs'], aggregator=torch.stack)
            # [b, cocoon_imgs, embed_dim]-->[b, embed_dim, cocoon_imgs]
            #scene_features = scene_features.permute(0, 2, 1).contiguous()
            #scene_features = self.sceneCocoonFuse(scene_features).squeeze()  # [b, 1, embed_dim]

        if self.train_vis_enc_only == False:
            obj_mmt_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(objects_pc_features)) + \
                         self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(batch['obj_offset']))
            batch['objectsGeo'] = batch['objectsGeo'].type(torch.FloatTensor).to(obj_mmt_in.device)
            if self.args.geo3d:
                obj_mmt_in = torch.cat((obj_mmt_in, batch['objectsGeo'][:, :, :24]), -1)
                obj_mmt_in = self.obj_3dgeofeat_feat_layer_norm(self.linear_obj_3dgeofeat_to_mmt_in(obj_mmt_in))

            if self.context_2d == 'aligned':
                obj_mmt_in = obj_mmt_in
                # + \
                # self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                # self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))

            obj_mmt_in = self.obj_drop(obj_mmt_in)
            obj_num = obj_mmt_in.size(1)
            obj_mask = _get_mask(batch['context_size'].to(obj_mmt_in.device), obj_num)  # all proposals are non-empty

            # Classify the segmented objects:
            if self.object_clf is not None and self.args.twoTrans==False:  # 3D branch
                objects_classifier_features = obj_mmt_in
                result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        if self.context_2d == 'unaligned':
            if self.args.context_info_2d_cached_file is not None:
                feat_2d = batch['feat_2d']
            else:
                feat_2d = objects_img_features

            if self.args.cocoon:
                batch['objectsGeo'] = batch['objectsGeo'][:, :, 0, :]
            if 'Geo' in (self.args.feat2d.replace('3D', '')):
                feat_2d = torch.cat((feat_2d, batch['objectsGeo']), -1)
            # geo_feat = self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['objectsGeo']))
            # feat_2d = torch.cat((feat_2d, geo_feat), -1)

            context_obj_mmt_in = self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(feat_2d))
            obj_num = context_obj_mmt_in.size(1)
            context_obj_mmt_in = self.context_drop(context_obj_mmt_in)
            context_obj_mask = _get_mask(batch['context_size'].to(context_obj_mmt_in.device),
                                         obj_num)  # all proposals are non-empty

            if self.args.twoTrans:
                mmt_vis_results = self.mmt_vis(obj_emb_3d=obj_mmt_in, obj_mask_3d=obj_mask,
                                               obj_emb_2d=context_obj_mmt_in, obj_mask_2d=context_obj_mask,
                                               obj_num=obj_num)
                obj_mmt_in = mmt_vis_results['mmt_obj3D_output']
                context_obj_mmt_in = mmt_vis_results['mmt_obj2D_output']
                # 3D cls
                if self.object_clf is not None:  # 3D branch
                    objects_classifier_features = obj_mmt_in
                    result['class_logits'] = get_siamese_features(self.object_clf,
                                                                  objects_classifier_features, torch.stack)
            else:
                obj_mmt_in = torch.cat([obj_mmt_in, context_obj_mmt_in], dim=1)  # concat 2D & 3D
                obj_mask = torch.cat([obj_mask, context_obj_mask], dim=1)

            if self.object_clf_2D is not None:  # 2D branch
                result['class_logits_2d'] = get_siamese_features(self.object_clf_2D, context_obj_mmt_in, torch.stack)  # [b, N_obj, N_cls]

            if self.train_vis_enc_only:
                return result

            if 'clspred' in (self.args.feat2d.replace('3D', '')):
                context_obj_mmt_in = torch.cat((context_obj_mmt_in, result['class_logits_2d']), -1)
                context_obj_mmt_in = self.obj2d_feat_layer_norm_clspred(self.linear_2d_feat_to_mmt_in_clspred(context_obj_mmt_in))

            if self.args.clspred3d:
                obj_mmt_in = torch.cat((obj_mmt_in, result['class_logits']), -1)
                obj_mmt_in = self.obj3d_feat_layer_norm_clspred(self.linear_3d_feat_to_mmt_in_clspred(obj_mmt_in))

        # Get feature for utterance
        txt_inds = batch["token_inds"]  # batch_size, lang_size
        txt_type_mask = torch.ones(txt_inds.shape, device=torch.device('cuda')) * 1.
        txt_mask = _get_mask(batch['token_num'].to(txt_inds.device), txt_inds.size(1))  # all proposals are non-empty
        txt_type_mask = txt_type_mask.long()
        text_bert_out = self.text_bert(
            txt_inds=txt_inds,
            txt_mask=txt_mask,
            txt_type_mask=txt_type_mask
        )
        txt_emb = self.text_bert_out_linear(text_bert_out)
        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(text_bert_out[:, 0, :])

        if self.args.twoTrans:
            mmt_results = self.mmt_vis_lang(txt_emb=txt_emb, txt_mask=txt_mask,
                                            obj_emb=obj_mmt_in, obj_mask=obj_mask,
                                            obj_num=obj_num)
            if self.args.sharetwoTrans:
                mmt_results_2D = self.mmt_vis_lang(txt_emb=txt_emb, txt_mask=txt_mask,
                                                   obj_emb=context_obj_mmt_in, obj_mask=context_obj_mask,
                                                   obj_num=obj_num)
            else:
                mmt_results_2D = self.mmt_vis_lang_2D(txt_emb=txt_emb, txt_mask=txt_mask,
                                                      obj_emb=context_obj_mmt_in, obj_mask=context_obj_mask,
                                                      obj_num=obj_num)
        else:
            mmt_results = self.mmt(
                txt_emb=txt_emb,
                txt_mask=txt_mask,
                obj_emb=obj_mmt_in,
                obj_mask=obj_mask,
                obj_num=obj_num
            )
        if self.args_mode == 'evaluate':
            assert (mmt_results['mmt_seq_output'].shape[1] == (self.text_length + obj_num))
        if self.args_mode != 'evaluate' and self.context_2d == 'unaligned':
            if self.args.twoTrans:
                assert (mmt_results['mmt_seq_output'].shape[1] == (self.text_length + obj_num))
            else:
                assert (mmt_results['mmt_seq_output'].shape[1] == (self.text_length + obj_num * 2))

        # 3D Referring:
        if self.args.twoTrans:
            result['logits3D_beforeFusion'] = self.matching_cls_3D_beforeFusion(mmt_results['mmt_obj_output'])
            result['logits2D_beforeFusion'] = self.matching_cls_2D_beforeFusion(mmt_results_2D['mmt_obj_output'])
            # Fusion:
            if self.fuse_conv1D:
                # Conv1D fusion
                mmt_results['mmt_obj_output'] = torch.stack((mmt_results['mmt_obj_output'],
                                                             mmt_results_2D['mmt_obj_output']), dim=-1)
                mmt_results['mmt_obj_output'] = get_siamese_features(self.ref_fusion3D_2D, mmt_results['mmt_obj_output'],
                                                                     aggregator=torch.stack).squeeze()
            else:
                # Transformer fusion
                if self.args.sceneCocoonPath:
                    mmt_results = self.ref_fusion3D_2D(obj_emb_3d=mmt_results['mmt_obj_output'], obj_mask_3d=obj_mask,
                                                       obj_emb_2d=mmt_results_2D['mmt_obj_output'],
                                                       obj_mask_2d=context_obj_mask, scene_emb=scene_features)
                else:
                    mmt_results = self.ref_fusion3D_2D(obj_emb_3d=mmt_results['mmt_obj_output'], obj_mask_3d=obj_mask,
                                                       obj_emb_2d=mmt_results_2D['mmt_obj_output'],
                                                       obj_mask_2d=context_obj_mask, obj_num=obj_num)
                mmt_results['mmt_obj_output'] = mmt_results['mmt_obj3D_output']
                mmt_results_2D['mmt_obj_output'] = mmt_results['mmt_obj2D_output']
            result['logits'] = self.matching_cls(mmt_results['mmt_obj_output'])  # Final 3D ref
        else:
            result['logits'] = self.matching_cls(mmt_results['mmt_obj_output'])  # Final 3D ref
        result['mmt_obj_output'] = mmt_results['mmt_obj_output']

        # 2D Referring:
        if self.context_2d == 'unaligned':
            if self.args.twoTrans:
                # Final 2D ref
                result['logits_2D'] = self.matching_cls_2D(mmt_results_2D['mmt_obj_output'])  # Final 2D ref
                result['mmt_obj_output_2D'] = mmt_results_2D['mmt_obj_output']
            else:
                result['logits_2D'] = self.matching_cls_2D(mmt_results['mmt_obj_output_2D'])  # Final 2D ref
                result['mmt_obj_output_2D'] = mmt_results['mmt_obj_output_2D']
        return result


def instantiate_referit3d_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # convenience
    geo_out_dim = args.object_latent_dim
    lang_out_dim = args.language_latent_dim
    mmt_out_dim = args.mmt_latent_dim  # 768

    # Optional, make a bbox encoder
    if args.obj_cls_alpha > 0:
        print('Adding an object-classification loss.')
        if args.softtripleloss:
            object_clf = object_decoder_for_clf(geo_out_dim, 128)  # 3D head
        else:
            object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)  # 3D head
        if args.context_2d == 'unaligned':
            if args.softtripleloss:
                object_clf_2D = object_decoder_for_clf(geo_out_dim, 128)  # 2D head
            else:
                object_clf_2D = object_decoder_for_clf(geo_out_dim, n_obj_classes)  # 2D head
        else:
            object_clf_2D = None
    else:
        object_clf = None
        object_clf_2D = None

    if args.model.startswith('mmt') and args.transformer:
        lang_out_dim = 768

    # Create Language Classification Head: 
    language_clf = None
    if args.lang_cls_alpha > 0:
        print('Adding a text-classification loss.')
        language_clf = text_decoder_for_clf(lang_out_dim, n_obj_classes)

    model = MMT_ReferIt3DNet(
        args=args,
        visudim=geo_out_dim,
        object_clf=object_clf,
        object_clf_2D=object_clf_2D,
        language_clf=language_clf,
        TEXT_BERT_HIDDEN_SIZE=lang_out_dim,
        MMT_HIDDEN_SIZE=mmt_out_dim,
        context_2d=args.context_2d,
        feat2dtype=args.feat2d,
        mmt_mask=args.mmt_mask,
        n_obj_classes=n_obj_classes)

    return model


# pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask
