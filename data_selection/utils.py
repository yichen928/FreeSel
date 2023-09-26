import torch
import os
import random
import math
import numpy as np
from collections import defaultdict, deque
import time
import torch.nn.functional as F
import datetime


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def get_distance(p1, p2, type, slice=1000):
    if len(p1.shape) > 1:
        if len(p2.shape) == 1:
            # p1 (n, dim)
            # p2 (dim)
            p2 = p2.unsqueeze(0)  # (1, dim)
            if type == "cosine":
                p1 = F.normalize(p1, p=2, dim=1)
                p2 = F.normalize(p2, p=2, dim=1)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = 1 - torch.sum(p1[slice*i:slice*(i+1)]*p2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            elif type == "euclidean":
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = torch.norm(p1[slice*i:slice*(i+1)]-p2, p=2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                raise NotImplementedError
        else:
            # p1 (n, dim)
            # p2 (m, dim)
            if type == "cosine":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                p2 = F.normalize(p2, p=2, dim=2)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    p1_slice = F.normalize(p1[slice*i:slice*(i+1)], p=2, dim=2)
                    dist_ = 1 - torch.sum(p1_slice * p2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)

            elif type == "euclidean":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = torch.norm(p1[slice*i:slice*(i+1)] - p2, p=2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                raise NotImplementedError
    else:
        # p1 (dim, )
        # p2 (dim, )
        if type == "cosine":
            dist = 1 - torch.sum(p1 * p2)
        elif type == "euclidean":
            dist = torch.norm(p1 - p2, p=2)
        else:
            raise NotImplementedError
    return dist


def update_distance(distances, all_features, cfeature, dist_func):
    new_dist = dist_func(all_features, cfeature)
    distances = torch.where(distances < new_dist, distances, new_dist)
    return distances


def update_distance_dense(distances, all_features, cfeatures, dist_func):
    # all_features: (n, c)
    # cfeatures: (r, c)
    new_dist = dist_func(all_features, cfeatures)  # (n, r)
    new_dist = torch.min(new_dist, dim=1)[0]  # (n, )
    distances = torch.where(distances < new_dist, distances, new_dist)
    return distances


def farthest_distance_sample(all_features, sample_num, dist_func, init_ids=[]):
    if len(init_ids) >= sample_num:
        print("Initial samples are enough")
        return init_ids

    if all_features.shape[0] <= sample_num:
        print("Not enough features")
        return list(range(all_features.shape[0]))

    total_num = all_features.shape[0]
    if len(init_ids) == 0:
        sample_ids = random.sample(range(total_num), 1)
    else:
        sample_ids = init_ids

    distances = torch.zeros(total_num).cuda() + 1e20

    for i, init_id in enumerate(sample_ids):
        distances = update_distance(distances, all_features, all_features[init_id], dist_func)

    while len(sample_ids) < sample_num:
        new_id = torch.max(distances, dim=0)[1]
        distances = update_distance(distances, all_features, all_features[new_id], dist_func)
        sample_ids.append(new_id.item())
        if len(sample_ids) % 100 == 1:
            print(len(sample_ids), "/", torch.max(distances, dim=0)[0], "FPS")
    assert len(set(sample_ids)) == sample_num
    return sample_ids


def farthest_distance_sample_dense(all_features, id2idx, sample_num, dist_func, init_ids=[], topk=None):
    if len(init_ids) >= sample_num:
        print("Initial samples are enough")
        return init_ids

    feature_num = all_features.shape[0]
    total_num = len(id2idx)
    if total_num <= sample_num:
        print("Not enough features")
        return list(range(total_num))

    idx2id = []
    for id in id2idx:
        idxs = id2idx[id]
        idx2id.extend([id]*idxs.shape[0])
    assert len(idx2id) == feature_num

    if len(init_ids) == 0:
        sample_ids = random.sample(range(total_num), 1)
    else:
        sample_ids = init_ids

    distances = torch.zeros(feature_num).cuda() + 1e20
    print(torch.max(distances, dim=0)[0])

    for i, init_id in enumerate(sample_ids):
        distances = update_distance_dense(distances, all_features, all_features[id2idx[init_id]], dist_func)
        if i % 100 == 1:
            print(i, torch.max(distances, dim=0)[0], "random")
            print(all_features.shape, all_features[id2idx[init_id]].shape)


    while len(sample_ids) < sample_num:
        new_featid = torch.max(distances, dim=0)[1]
        new_id = idx2id[new_featid]
        distances = update_distance_dense(distances, all_features, all_features[id2idx[new_id]], dist_func)
        sample_ids.append(new_id)
        if len(sample_ids) % 100 == 1:
            print(len(sample_ids))
            print(len(sample_ids), torch.max(distances, dim=0)[0], "FDS")
            print(all_features.shape, all_features[id2idx[new_id]].shape)
    assert len(set(sample_ids)) == sample_num
    return sample_ids


def prob_seed_dense(all_features, id2idx, sample_num, dist_func, init_ids=[]):
    if len(init_ids) >= sample_num:
        print("Initial samples are enough")
        return init_ids

    feature_num = all_features.shape[0]
    total_num = len(id2idx)
    if total_num <= sample_num:
        print("Not enough features")
        return list(range(total_num))

    idx2id = []
    for id in id2idx:
        idxs = id2idx[id]
        idx2id.extend([id]*idxs.shape[0])
    assert len(idx2id) == feature_num

    if len(init_ids) == 0:
        sample_ids = random.sample(range(total_num), 1)
    else:
        sample_ids = init_ids

    distances = torch.zeros(feature_num).cuda() + 1e20
    print(torch.max(distances, dim=0)[0])

    for i, init_id in enumerate(sample_ids):
        distances = update_distance_dense(distances, all_features, all_features[id2idx[init_id]], dist_func)
        if i % 100 == 1:
            print(i, torch.max(distances, dim=0)[0], "random")
            print(all_features.shape, all_features[id2idx[init_id]].shape)

    while len(sample_ids) < sample_num:
        prob = distances ** 2 / torch.sum(distances ** 2)
        prob = prob.cpu().numpy()
        new_featid = np.random.choice(distances.shape[0], p=prob)

        # new_featid = torch.max(distances, dim=0)[1]
        new_id = idx2id[new_featid]
        distances = update_distance_dense(distances, all_features, all_features[id2idx[new_id]], dist_func)
        sample_ids.append(new_id)
        if len(sample_ids) % 100 == 1:
            print(len(sample_ids))
            print(len(sample_ids), torch.max(distances, dim=0)[0], "prob")
            print(all_features.shape, all_features[id2idx[new_id]].shape)
    assert len(set(sample_ids)) == sample_num
    return sample_ids
