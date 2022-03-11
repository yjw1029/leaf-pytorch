from collections import OrderedDict
from platformdirs import user_runtime_dir
import torch

def none_uniform(updates):
    avg_param = OrderedDict()
    total_weight = 0.
    for (client_samples, client_model) in updates:
        total_weight += client_samples
        for name, param in client_model.items():
            if name not in avg_param:
                avg_param[name] = client_samples * param
            else:
                avg_param[name] += client_samples * param

    for name in avg_param:
        avg_param[name] = avg_param[name] / total_weight
    return avg_param

def uniform(updates):
    avg_param = OrderedDict()
    total_weight = 0.
    for (_, client_model) in updates:
        total_weight += 1.
        for name, param in client_model.items():
            if name not in avg_param:
                avg_param[name] = param
            else:
                avg_param[name] += param

    for name in avg_param:
        avg_param[name] = avg_param[name] / total_weight
    return avg_param

def krum(updates, global_model, krum_mal_num):
    avg_param = OrderedDict()
    total_weight = 0.
    for (_, client_model) in updates:
        total_weight += 1.
        for name, param in client_model.items():
            if name not in avg_param:
                avg_param[name] = [param - global_model[name]]
            else:
                avg_param[name].append(param - global_model[name])

    for name in avg_param:
        avg_param[name] = torch.stack(avg_param[name], dim=0)

    with torch.no_grad():
        user_num = len(updates)
        user_flatten_grad = []
        for u_i in range(user_num):
            user_flatten_grad_i = []
            for name in avg_param:
                user_flatten_grad_i.append(torch.flatten(avg_param[name][u_i]))
            user_flatten_grad_i = torch.cat(user_flatten_grad_i)
            user_flatten_grad.append(user_flatten_grad_i)
        user_flatten_grad = torch.stack(user_flatten_grad)

        # compute l2 distance between users
        user_scores = torch.zeros((user_num, user_num), device=user_flatten_grad.device)
        for u_i in range(user_num):
            user_scores[u_i] = torch.norm(
                user_flatten_grad - user_flatten_grad[u_i],
                dim=list(range(len(user_flatten_grad.shape)))[1:],
            )
            user_scores[u_i, u_i] = torch.inf

        # select summation od smallest n-f-2 scores
        topk_user_scores, _ = torch.topk(
            user_scores, k=user_num - krum_mal_num - 2, dim=1, largest=False
        )
        sm_user_scores = torch.sum(topk_user_scores, dim=1)

        # users with smallest score is selected as update gradient
        u_score, select_u = torch.topk(sm_user_scores, k=1, largest=False)
        select_u = select_u[0]
        
    krum_grad = OrderedDict()
    for name in avg_param:
        krum_grad[name] = avg_param[name][select_u] + global_model[name]
    return krum_grad

def multi_krum(updates, global_model,krum_mal_num, multi_krum_num):
    avg_param = OrderedDict()
    total_weight = 0.
    for (_, client_model) in updates:
        total_weight += 1.
        for name, param in client_model.items():
            if name not in avg_param:
                avg_param[name] = [param - global_model[name]]
            else:
                avg_param[name].append(param - global_model[name])

    for name in avg_param:
        avg_param[name] = torch.stack(avg_param[name], dim=0)

    with torch.no_grad():
        user_num = len(updates)
        user_flatten_grad = []
        for u_i in range(user_num):
            user_flatten_grad_i = []
            for name in avg_param:
                user_flatten_grad_i.append(torch.flatten(avg_param[name][u_i]))
            user_flatten_grad_i = torch.cat(user_flatten_grad_i)
            user_flatten_grad.append(user_flatten_grad_i)
        user_flatten_grad = torch.stack(user_flatten_grad)

        # compute l2 distance between users
        user_scores = torch.zeros((user_num, user_num), device=user_flatten_grad.device)
        for u_i in range(user_num):
            user_scores[u_i] = torch.norm(
                user_flatten_grad - user_flatten_grad[u_i],
                dim=list(range(len(user_flatten_grad.shape)))[1:],
            )
            user_scores[u_i, u_i] = torch.inf

        # select summation od smallest n-f-2 scores
        topk_user_scores, _ = torch.topk(
            user_scores, k=user_num - krum_mal_num - 2, dim=1, largest=False
        )
        sm_user_scores = torch.sum(topk_user_scores, dim=1)

        # users with smallest score is selected as update gradient
        u_score, select_u = torch.topk(sm_user_scores, k=multi_krum_num, largest=False)
        
    krum_grad = OrderedDict()
    for name in avg_param:
        krum_grad[name] = torch.mean(avg_param[name][select_u], dim=0) + global_model[name]
    return krum_grad

def median(updates):
    median_param = OrderedDict()
    for (_, client_model) in updates:
        for name, param in client_model.items():
            if name not in median_param:
                median_param[name] = [param]
            else:
                median_param[name].append(param)

    for name in median_param:
        median_param[name] = torch.median(
            torch.stack(median_param[name], dim=0), dim=0)
    return median_param

def trimmed_mean(updates, k):
    avg_param = OrderedDict()
    for (_, client_model) in updates:
        for name, param in client_model.items():
            if name not in avg_param:
                avg_param[name] = [param]
            else:
                avg_param[name].append(param)

    for name in avg_param:
        avg_param[name] = torch.stack(avg_param[name], dim=0)
        user_num = avg_param[name].size(0)
        largest_value, _ = torch.topk(avg_param[name], k=k, dim=0)
        smallest_value, _ = torch.topk(avg_param[name], k=k, dim=0, largest=False)

        result = (
            torch.sum(avg_param[name], dim=0)
            - torch.sum(largest_value, dim=0)
            - torch.sum(smallest_value, dim=0)
        ) / (user_num - 2 * k)

        avg_param[name] = result
    return avg_param

def norm_bound(updates, global_model, M):
    pass
