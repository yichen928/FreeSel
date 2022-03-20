import os
import sys
import argparse
import pickle
import cv2

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import torch.nn.functional as F
from PIL import Image, ImageFile

import utils
import vision_transformer as vits
import torchextractor as tx
from kmeans_pytorch import kmeans
import random
from functools import partial

from dataset import VOCReturnIndexDataset


def extract_feature_pipeline(args):
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](pretrained=False)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    all_features = {}
    for year in ["2007", "2012"]:
        dataset_train = VOCReturnIndexDataset(args.data_path, year, transform=transform)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        print(f"Data loaded with {len(dataset_train)} trainimgs.")

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features, train_ids = extract_features(model, data_loader_train, args)

        for i in range(len(train_features)):
            all_features[train_ids[i]] = train_features[i]

    # save features and labels
    if args.dump_features:
        with open(args.dump_features, "wb") as file:
            pickle.dump(all_features, file)

    return all_features


def filter_features(dense_features, args, attn=None):
    # input: (n, c, k, k)
    # output: list n: [c1, c2, ...,]
    filtered_features = []
    count = 0

    bs = dense_features.shape[0]
    if "vit" not in args.arch:
        dense_features = dense_features.permute(0, 2, 3, 1)
        dense_features = dense_features.reshape(bs, dense_features.shape[1]*dense_features.shape[2], dense_features.shape[3])  # (n, k*k,c )

    dense_features_norm = torch.norm(dense_features, p=2, dim=2)  # (n, k*k)

    if attn is None:
        mask = dense_features_norm > args.threshold # (n, k*k)
    else:
        assert 0 <= args.threshold <= 1
        # attn: (bs, wh)
        attn_sort, idx_sort = torch.sort(attn, dim=1, descending=False)
        attn_cum = torch.cumsum(attn_sort, dim=1)  # (bs, wh)
        mask = attn_cum > (1-args.threshold)
        for b in range(bs):
            mask[b][idx_sort[b]] = mask[b].clone()

    for b in range(bs):
        mask_i = mask[b]  # (k*k, )
        dense_features_i = dense_features[b]  # (k*k, c)
        if torch.sum(mask_i) > 0:
            dense_features_i = dense_features_i[mask_i]
        else:
            max_id = torch.max(dense_features_norm[b], dim=0)[1]
            dense_features_i = dense_features_i[max_id].unsqueeze(0)  # (1, c)

        if args.centroid_num is not None and args.centroid_num < dense_features_i.shape[0]:
            cluster_ids_x, cluster_centers = kmeans(
                X=dense_features_i, num_clusters=args.centroid_num, distance=args.kmeans_dist_type, iter_limit=100, device=torch.device('cuda:0')
            )
            count += cluster_centers.shape[0]
            filtered_features.append(cluster_centers.cuda())
        else:
            filtered_features.append(dense_features_i)
            count += dense_features_i.shape[0]

    return filtered_features, count


@torch.no_grad()
def extract_features(model, data_loader, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    if "vit" not in args.arch:
        model = tx.Extractor(model, ["layer4"])

    train_ids = []
    train_features = []
    feature_num = 0
    for samples, index in metric_logger.log_every(data_loader, 1):
        samples = samples.cuda(non_blocking=True)
        if "vit" not in args.arch:
            feats, dense_feats = model(samples)
            dense_feats = dense_feats["layer4"].clone()
            dense_feats, count = filter_features(dense_feats, args)
        else:
            dense_feats = model.get_intermediate_layers(samples, n=2)
            dense_feats = dense_feats[0]
            dense_feats = dense_feats[:, 1:]
            attn = model.get_last_selfattention(samples)  # (bs, nh, wh+1, wh+1)
            attn = torch.mean(attn, dim=1)[:, 0, 1:]  # (bs, wh)
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            dense_feats, count = filter_features(dense_feats, args, attn)

        feature_num += count

        train_features.extend(dense_feats)
        train_ids.extend(index)

    return train_features, train_ids


def merge_features(all_features):
    merged_features = list(all_features.values())
    merged_features = torch.cat(merged_features, dim=0)
    id2idx = {}
    idx = 0
    count = 0
    merged_ids = []
    for id in all_features:
        id2idx[count] = torch.arange(idx, idx+all_features[id].shape[0])
        merged_ids.append(id)
        idx = idx + all_features[id].shape[0]
        count += 1
    return merged_features, merged_ids, id2idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('save the extracted features')
    parser.add_argument('--batch_size_per_gpu', default=24, type=int, help='Per-GPU batch-size')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--pretrained_weights', default='pretrain/dino_resnet50_pretrain_full_checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features',
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--data_path', default='data', type=str, help='path for the PSCAL VOC dataset')
    parser.add_argument('--threshold', default=0.5, type=float, help='the attention ratio')
    parser.add_argument('--centroid_num', default=5, type=int, help='number of kmeans centers')
    parser.add_argument('--dist_type', type=str, default="cosine", help="cosine or euclidean for K-center-greedy")
    parser.add_argument('--kmeans_dist_type', type=str, default="euclidean", help="cosine or euclidean for kmeans")
    parser.add_argument('--sampling', type=str, default="prob", choices=["prob", "FDS"], help="strategy for sampling")
    parser.add_argument('--selected_num', type=int, default=3000, help="selected sample number")
    parser.add_argument('--random_num', type=int, default=None, help="randomly select some samples")
    parser.add_argument('--save_name', type=str, default="dino_vits16_thre5e-1_kmeans5_cosine_prob")
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    print("Extract features from images...")
    train_features = extract_feature_pipeline(args)
    merged_features, merged_ids, id2idx = merge_features(train_features)

    print("Select Samples...")
    if args.random_num is None:
        if args.sampling == "prob":
            selected_sample = utils.prob_seed_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type))
        else:
            selected_sample = utils.farthest_distance_sample_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type))
    else:
        init_ids = random.sample(range(len(id2idx)), args.random_num)
        if args.sampling == "prob":
            selected_sample = utils.prob_seed_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type), init_ids=init_ids)
        else:
            selected_sample = utils.farthest_distance_sample_dense(merged_features, id2idx, args.selected_num, partial(utils.get_distance, type=args.dist_type), init_ids=init_ids)

    samples_07 = []
    samples_12 = []
    for idx in selected_sample:
        id = merged_ids[idx]
        if "_" in id:
            samples_12.append(id+"\n")
        else:
            samples_07.append(id+"\n")

    samples_07.sort()
    samples_12.sort()
    filename = "trainval_%s_%d.txt"%(args.save_name, args.selected_num)
    with open(os.path.join(args.data_path, "VOCdevkit/VOC2007/ImageSets/Main", filename), "w") as file:
        file.writelines(samples_07)
    with open(os.path.join(args.data_path, "VOCdevkit/VOC2012/ImageSets/Main", filename), "w") as file:
        file.writelines(samples_12)