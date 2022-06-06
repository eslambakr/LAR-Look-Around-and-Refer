#!/usr/bin/env python
# coding: utf-8
"""
1) To run SAT 2d Features:
--patience 100 --max-train-epochs 100 --init-lr 1e-4 --batch-size 2 --n-workers 2 --transformer --model mmt_referIt3DNet -scannet-file /home/e/scannet_dataset/scannet/scan_4_nr3d/keep_all_points_00_view_no_global_scan_alignment_saveJPG_cocoon_twoStreams.pkl -referit3D-file /home/e/scannet_dataset/scannet/nr3d.csv --log-dir ../del --unit-sphere-norm True --feat2d ROI --context_2d unaligned --mmt_mask train2d --warmup -load-imgs False --img-encoder False --object-encoder pnet_pp -load-dense False --train-vis-enc-only False --imgsize 32 --cocoon False --twoStreams False --context_info_2d_cached_file /media/e/0d208863-5cdb-4a43-9794-3ca8726831b3/3D_visual_grounding/dataset/scannet
2) To run our 2d Features (clssonly):
--patience 100 --max-train-epochs 100 --init-lr 1e-4 --batch-size 2 --n-workers 2 --transformer --model mmt_referIt3DNet -scannet-file /home/e/scannet_dataset/scannet/scan_4_nr3d/keep_all_points_00_view_no_global_scan_alignment_saveJPG_cocoon_twoStreams.pkl -referit3D-file /home/e/scannet_dataset/scannet/nr3d.csv --log-dir ../del --unit-sphere-norm True --feat2d clsvecROI --context_2d unaligned --mmt_mask train2d --warmup -load-imgs True --img-encoder True --object-encoder convnext -load-dense False --train-vis-enc-only True --imgsize 32 --cocoon False --twoStreams False
3) To run our 2d features (E2E):
-scannet-file /home/e/scannet_dataset/scannet/scan_4_nr3d/keep_all_points_00_view_no_global_scan_alignment_saveJPG_cocoon_twoStreams/keep_all_points_00_view_no_global_scan_alignment_saveJPG_cocoon_twoStreams.pkl -referit3D-file /home/e/scannet_dataset/scannet/nr3d.csv --log-dir ../log/referit3d_r18_32_loadimgs --n-workers 2 --batch-size 2 -load-imgs True --img-encoder True --object-encoder convnext_p++ -load-dense False --train-vis-enc-only False --cocoon False --twoStreams True --obj-cls-alpha 5 --unit-sphere-norm True --feat2d ROIGeoclspred --context_2d unaligned --mmt_mask train2d --warmup --transformer --model mmt_referIt3DNet --twoTrans True --sharetwoTrans False --tripleloss False --init-lr 0.0001 --feat2ddim 2048 --imgsize 32
"""
import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored
import torch.multiprocessing as mp
import torch.distributed as dist
from referit3d.losses.SoftTriple import SoftTriple
import logging
# ## might be related to the memory issue https://github.com/referit3d/referit3d/issues/5
# ## A temp solution is to add at evlauation mode to avoid "received 0 items of ancdata" (uncomment next line in eval)
# torch.multiprocessing.set_sharing_strategy('file_system')

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import instantiate_referit3d_net
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from referit3d.utils.scheduler import GradualWarmupScheduler


def log_train_test_information(args, epoch, logger, train_meters, test_meters, timings, best_test_acc, best_test_epoch):
    """
    Helper logging function.
    """
    logger.info('Epoch:{}'.format(epoch))
    for phase in ['train', 'test']:
        if phase == 'train':
            meters = train_meters
        else:
            meters = test_meters

        info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                    meters[phase + '_total_loss'],
                                                                    meters[phase + '_referential_acc'])

        if args.obj_cls_alpha > 0:
            info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

        if args.lang_cls_alpha > 0:
            info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

        logger.info(info)
        logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
    logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Read the scan related information
    print("starting caching the pkl files.....")
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file, load_dense=args.load_dense,
                                                                          args=args)
    print("Finish caching the pkl files, Done")

    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)

    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    gen = torch.Generator()  # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    data_loaders, samplers = make_data_loaders(args, referit_data, vocab, class_to_idx,
                                               all_scans_in_dict, mean_rgb, gen)
    # Prepare GPU environment
    # set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"
    if args.distributed:
        device = torch.device(args.gpu)
    else:
        device = torch.device('cuda')
    seed_training_code(args.random_seed, gen=gen)

    # Losses:
    criteria = dict()

    # Referential, "find the object in the scan" loss
    if args.s_vs_n_weight is not None:  # TODO - move to a better place
        assert args.augment_with_sr3d is not None
        ce = nn.CrossEntropyLoss(reduction='none').to(device)
        s_vs_n_weight = args.s_vs_n_weight

        def weighted_ce(logits, batch):
            loss_per_example = ce(logits, batch['target_pos'])
            sr3d_mask = ~batch['is_nr3d']
            weights = torch.ones(loss_per_example.shape).to(device)
            weights[sr3d_mask] = s_vs_n_weight
            loss_per_example = loss_per_example * weights
            loss = loss_per_example.sum() / len(loss_per_example)
            return loss

        criteria['logits'] = weighted_ce
    else:
        criteria['logits'] = nn.CrossEntropyLoss().to(device)  # 3D Ref Loss
        if args.twoTrans:
            criteria['logits_early'] = nn.CrossEntropyLoss().to(device)  # 3D Early Ref Loss
    criteria['logits_nondec'] = nn.CrossEntropyLoss(reduction='none').to(device)

    # Contrastive loss between target and distractors:
    if args.contrastiveloss:
        from referit3d.losses.contrastive_loss import ContrastiveLoss
        criteria['contrastiveloss_3d'] = ContrastiveLoss(margin=3)
        criteria['contrastiveloss_2d'] = ContrastiveLoss(margin=3)

    # Object-type classification
    if args.obj_cls_alpha > 0:
        reduction = 'mean' if args.s_vs_n_weight is None else 'none'
        if args.train_vis_enc_only == False:
            if args.softtripleloss:
                criteria['class_logits'] = SoftTriple(20, 0.1, 0.2, 0.01, dim=128, cN=len(class_to_idx), K=10,
                                                      class_to_ignore=class_to_idx['pad'],
                                                      reduction=reduction, device=device).to(device)
            else:
                criteria['class_logits'] = nn.CrossEntropyLoss(ignore_index=class_to_idx['pad'],
                                                               reduction=reduction).to(device)

        if args.context_2d == 'unaligned':
            criteria['logits_2d'] = nn.CrossEntropyLoss().to(device)  # 2D Ref Loss
            if args.twoTrans:
                criteria['logits_2d_early'] = nn.CrossEntropyLoss().to(device)  # 2D Early Ref Loss
            if args.softtripleloss:
                criteria['class_logits_2d'] = SoftTriple(20, 0.1, 0.2, 0.01, dim=128, cN=len(class_to_idx), K=10,
                                                         class_to_ignore=class_to_idx['pad'],
                                                         reduction=reduction, device=device).to(device)
            else:
                criteria['class_logits_2d'] = nn.CrossEntropyLoss(ignore_index=class_to_idx['pad'],
                                                                  reduction=reduction).to(device)

    # Target-in-language guessing
    if args.lang_cls_alpha > 0 and (args.train_vis_enc_only == False):
        reduction = 'mean' if args.s_vs_n_weight is None else 'none'
        criteria['lang_logits'] = nn.CrossEntropyLoss(reduction=reduction).to(device)

    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']

    model = instantiate_referit3d_net(args, vocab, n_classes).to(device)
    same_backbone_lr = False
    if same_backbone_lr:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    else:
        backbone_name = []
        if args.transformer:
            backbone_name.append('text_bert.')  # exclude text_bert_out_linear
            # backbone_name.append('object_encoder.')
            # backbone_name.append('cnt_object_encoder.')
        backbone_param, rest_param = [], []
        for kv in model.named_parameters():
            isbackbone = [int(key in kv[0]) for key in backbone_name]
            if sum(isbackbone + [0]):
                backbone_param.append(kv[1])
            else:
                rest_param.append(kv[1])
        optimizer = optim.Adam([{'params': rest_param},
                                {'params': backbone_param, 'lr': args.init_lr / 10.}], lr=args.init_lr)

        sum_backbone = sum([param.nelement() for param in backbone_param])
        sum_fusion = sum([param.nelement() for param in rest_param])
        sum_all = sum([param.nelement() for param in model.parameters()])
        print('backbone, fusion module parameters:', sum_backbone, sum_fusion, sum_all)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65,
                                                              patience=5, verbose=True)
    if args.patience == args.max_train_epochs:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40, 50, 60, 70, 80, 90],
                                                            gamma=0.65)  # custom2
        if args.max_train_epochs == 120: lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                                             milestones=[25, 40, 55, 70,
                                                                                                         85, 100],
                                                                                             gamma=0.5)  # custom3-120ep
    if args.warmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)
        optimizer.zero_grad()  # this zero gradient update is needed to avoid a warning message, issue #8.
        optimizer.step()

    # We must freeze the weights before DDP. https://github.com/open-mmlab/mmdetection/issues/2153
    if args.pretrained_path is not None:
        # This feature act as a warmup epochs to load the visual encoders after trained alone.
        print("Loading pretrained visual encoder weights...")
        load_state_dicts(args.pretrained_path, map_location=device, strict=False, args=args, model=model)
        print("Visual encoder loaded!")
        #print("Before Loading = ", model.object_encoder[0].stages[0].blocks[0].conv_dw.weight)

        if args.freeze_backbone:
            print("The backbone is Freezed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Freezing the 2d backbone:
            if args.multiprocessing_distributed and False:
                freezed_part = model.module
            else:
                freezed_part = model
            freezed_part = freezed_part.img_object_encoder
            for k, v in freezed_part.named_parameters():
                v.requires_grad = False

    # Distributed Training:
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.twoStreams and False:
                for module in model.object_encoder:
                    module.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # https://towardsdatascience.com/distributed-neural-network-training-in-pytorch-5e766e2a9e62
            # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
            # https://discuss.pytorch.org/t/save-model-for-distributeddataparallel/47129/7
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            if args.twoStreams and False:
                for module in model.object_encoder:
                    module.cuda()

            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.twoStreams and False:
            for module in model.object_encoder:
                module.cuda(args.gpu)
    else:
        torch.cuda.set_device(0)
        model = model.cuda(0)
        if args.twoStreams and False:
            for module in model.object_encoder:
                module.cuda(0)
    torch.backends.cudnn.benchmark = True

    start_training_epoch = 1
    best_test_acc = -1
    best_test_objClass_acc = -1
    best_test_epoch = -1
    no_improvement = 0

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model, args=args)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            best_test_epoch = loaded_epoch
            best_test_acc = lr_scheduler.best
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    if args.eval_path:
        load_model = torch.load(args.eval_path, map_location=device)
        print("Loaded Epoch is:", load_model['epoch'])
        if args.multiprocessing_distributed:
            model.load_state_dict(load_model['model'], strict=True)
        else:
            pretrained_dict = load_model['model']
            pretrained_dict = {key.replace("module.", ''): item for key, item in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=True)
        print("=> loaded pretrain model at {}".format(args.eval_path))
        if 'best' in load_model['lr_scheduler']:
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                load_model['lr_scheduler']['best']))

    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir)
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')
        eval_acc = 0.

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                # Train:
                # epoch=epoch,
                if args.warmup:
                    scheduler_warmup.step(metrics=eval_acc)  # using the previous epoch's metrics
                # print('lr:', epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])

                if args.distributed:
                    samplers['train'].set_epoch(epoch)

                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                if not args.multiprocessing_distributed or (
                        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    # Evaluate:
                    tic = time.time()
                    test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
                    toc = time.time()
                    timings['test'] = (toc - tic) / 60

                    eval_acc = test_meters['test_referential_acc']
                    if not args.warmup:
                        lr_scheduler.step(epoch=epoch, metrics=eval_acc)
                    # else: lr_scheduler.step(eval_acc)

                    if best_test_objClass_acc < test_meters['test_object_cls_acc']:
                        logger.info(colored('Object Class Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                        best_test_objClass_acc = test_meters['test_object_cls_acc']
                        save_state_dicts(osp.join(args.checkpoint_dir, 'best_objClass_model.pth'),
                                         epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

                    if best_test_acc < eval_acc:
                        logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                        best_test_acc = eval_acc
                        best_test_epoch = epoch

                        # Save the model (overwrite the best one)
                        save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                         epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                    log_train_test_information(args, epoch, logger, train_meters, test_meters, timings,
                                               best_test_acc, best_test_epoch)
                    train_meters.update(test_meters)
                    train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                                          main_tag='acc')
                    train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                                          step=epoch, main_tag='loss')
                    bar.refresh()

                    if no_improvement == args.patience:
                        logger.warning(
                            colored('Stopping the training @epoch-{} due to lack of progress in test-accuracy '
                                    'boost (patience hit {} epochs)'.format(epoch, args.patience),
                                    'red', attrs=['bold', 'underline']))
                        break

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            msg = ('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch))
            f_out.write(msg)

        logger.info('Finished training successfully. Good job!')

    elif args.mode == 'evaluate':
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            meters = evaluate_on_dataset(model, data_loaders['train'], criteria, device, pad_idx, args=args)
            print('Training Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
            print('Training Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
            print('Training Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))
            print('Training Reference-Accuracy 2D: {:.4f}'.format(meters['test_referential_acc_2d']))
            print('Training Object-Clf-Accuracy 2D: {:.4f}'.format(meters['test_object_cls_acc_2d']))
            print("-------------------------------------------------------------------")
            meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args)
            print('Testing Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
            print('Testing Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
            print('Testing Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))
            print('Testing Reference-Accuracy 2D: {:.4f}'.format(meters['test_referential_acc_2d']))
            print('Testing Object-Clf-Accuracy 2D: {:.4f}'.format(meters['test_object_cls_acc_2d']))
            print("-------------------------------------------------------------------")

            out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
            res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                      args, out_file=out_file)
            print(res)


if __name__ == '__main__':
    # print pytorch info:
    print(torch.__version__)
    print(" Number of available GPUs is: ", torch.cuda.device_count())
    print(torch.cuda.is_available())

    # Parse arguments
    args = parse_arguments()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
