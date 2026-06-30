import random
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import math
import os

import torch
import torch.nn.functional as F
import operator
import torch.nn as nn
from info_nce import InfoNCE
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import numpy as np

import clip
from utils import *
import open_clip


BPRE_POSITIVE_KEYS = [
    'enabled',
    'shot_capacity',
    'alpha',
    'beta',
    'temperature',
    'gamma',
    'momentum',
    'update_interval',
    'threshold',
    'warmup_steps',
    'min_reward',
    'lambda_sim',
    'lambda_conf',
    'lambda_div',
    'align_temperature',
    'prototype_logit_scale',
    'global_update_threshold',
    'steps',
    'optimizer_eps',
    'weight_decay',
    'log_interval',
]


def validate_positive_config(pos_cfg):
    if pos_cfg is None:
        raise ValueError('Missing positive configuration section.')
    missing = [key for key in BPRE_POSITIVE_KEYS if key not in pos_cfg]
    if missing:
        raise ValueError(f'Missing positive configuration keys: {", ".join(missing)}')
    return pos_cfg

def setup_logging(dataset_name):
    """Setup logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'BPRE_{dataset_name}_{timestamp}.log')
    
    logger = logging.getLogger(f'BPRE_{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class PrototypeManager:
    def __init__(self, num_classes, feature_dim, momentum, update_interval, init_prototypes=None):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.update_interval = update_interval

        if init_prototypes is None:
            self.text_prototypes = torch.zeros(num_classes, feature_dim).cuda().half()
            self.visual_prototypes = torch.zeros(num_classes, feature_dim).cuda().half()
        else:
            init_prototypes = F.normalize(init_prototypes.detach(), dim=1).cuda().half()
            self.text_prototypes = init_prototypes.clone()
            self.visual_prototypes = init_prototypes.clone()

        self.update_counts = torch.zeros(num_classes).cuda()
        self.quality_scores = torch.zeros(num_classes).cuda()

    def update_from_cache(self, cache, step, temperature):
        if step % self.update_interval != 0:
            return

        with torch.no_grad():
            for class_index, items in cache.items():
                if len(items) == 0:
                    continue

                features = torch.cat([item[0] for item in items], dim=0)
                rewards = torch.tensor(
                    [item[2] for item in items],
                    device=features.device,
                    dtype=features.dtype,
                )
                weights = F.softmax(rewards / temperature, dim=0)
                weighted_feature = (features * weights.unsqueeze(1)).sum(0)

                old_proto = self.visual_prototypes[class_index]
                new_proto = self.momentum * old_proto + (1 - self.momentum) * weighted_feature
                self.visual_prototypes[class_index] = F.normalize(new_proto, dim=0)

                self.update_counts[class_index] += 1
                self.quality_scores[class_index] = rewards.mean()
    
    def get_prototypes(self, class_ids=None):
        if class_ids is None:
            return self.text_prototypes, self.visual_prototypes
        return (self.text_prototypes[class_ids], self.visual_prototypes[class_ids])
    
    def get_quality_stats(self):
        return {
            'update_counts': self.update_counts.cpu().numpy(),
            'quality_scores': self.quality_scores.cpu().numpy()
        }

class AdaptiveRewardCalculator:
    def __init__(
        self,
        temperature,
        warmup_steps,
        min_reward,
        lambda_sim,
        lambda_conf,
        lambda_div,
        history_size,
    ):
        self.temperature = temperature
        self.warmup_steps = warmup_steps
        self.min_reward = min_reward
        self.lambda_sim = lambda_sim
        self.lambda_conf = lambda_conf
        self.lambda_div = lambda_div
        self.history_size = history_size
        
        self.step = 0
        self.reward_history = []

    def compute_reward(self, features, logits, prototypes, labels=None):
        warmup_factor = min(self.step / self.warmup_steps, 1.0)
        
        similarity = F.cosine_similarity(
            features.unsqueeze(1),
            prototypes.unsqueeze(0),
            dim=2
        )

        probs = F.softmax(logits / self.temperature, dim=1)
        confidence = 1.0 - (
            -(probs * torch.log(probs + 1e-10)).sum(1)
            / math.log(probs.size(1))
        )
        
        if len(self.reward_history) > 0:
            history = torch.stack(self.reward_history[-self.history_size:])
            diversity = 1.0 - torch.max(
                F.cosine_similarity(
                    features.unsqueeze(1),
                    history.unsqueeze(0),
                    dim=2
                ),
                dim=1
            ).values
        else:
            diversity = torch.ones_like(confidence)
            
        reward = (
            similarity.mean(1) * self.lambda_sim +
            confidence * self.lambda_conf +
            diversity * self.lambda_div
        )
        
        reward = self.min_reward + (reward - self.min_reward) * warmup_factor
        
        if features.size(0) > 0:
            self.reward_history.append(features.mean(0))
        if len(self.reward_history) > self.history_size:
            self.reward_history.pop(0)
            
        self.step += 1
        
        return reward
        
    def reset_history(self):
        self.reward_history = []
        self.step = 0

class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights
    
    def reset(self):
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)

class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), requires_grad=True)
        
    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys

def InfoNCELoss(A, B, temperature):
    loss = InfoNCE(temperature=temperature, reduction='mean')
    return loss(A, B)

def update_cache(cache, pred, feature_entry, shot_capacity):
    with torch.no_grad():
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(feature_entry)
            elif feature_entry[2] > cache[pred][-1][2]:
                cache[pred][-1] = feature_entry
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(2), reverse=True)
        else:
            cache[pred] = [feature_entry]
        return

def cache_key_value(image_features, cache, alpha, beta, clip_weights, reward_temperature):
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        
        for class_index in sorted(cache.keys()):
            items = cache[class_index]
            features = torch.cat([item[0] for item in items], dim=0)
            rewards = torch.tensor(
                [item[2] for item in items],
                device=features.device,
                dtype=features.dtype,
            )
            weights = F.softmax(rewards / reward_temperature, dim=0)
            image_prototype = (features * weights.unsqueeze(1)).sum(0, keepdim=True)
            image_prototype = F.normalize(image_prototype, dim=1)
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = F.one_hot(
            torch.tensor(cache_values).to(torch.int64),
            num_classes=clip_weights.size(1)
        ).cuda().half()
            
        return cache_keys, cache_values, all_classes
    
def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits

def compute_prototype_logits(image_features, text_prototypes, visual_prototypes, gamma, logit_scale):
    text_prototypes = F.normalize(text_prototypes, dim=1)
    visual_prototypes = F.normalize(visual_prototypes, dim=1)
    prototype_residual = visual_prototypes - text_prototypes
    return gamma * logit_scale * image_features @ prototype_residual.T

def visualize_cache(cache, iter):
    with torch.no_grad():
        cache_features = []
        cache_labels = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_features.append(item[0].reshape(-1))
                cache_labels.append(class_index)
                
        if len(cache_features) == 0:
            return
                
        cache_features = torch.stack(cache_features, dim=0)
        cache_labels = torch.tensor(cache_labels).to(torch.int64)
        cache_features = F.normalize(cache_features, dim=1)
        cache_features = cache_features.cpu().numpy()
        cache_labels = cache_labels.cpu().numpy()
        
        n_samples = cache_features.shape[0]
        perplexity = min(30, n_samples - 1)
        
        if n_samples < 4:
            return
            
        tsne = TSNE(n_components=2, perplexity=perplexity)
        cache_features_fit = tsne.fit_transform(cache_features)
        
        colors = [
            '#00429d', '#93003a', '#007d34', '#ff6800', '#e30022',
            '#a6bdd7', '#ffcc00', '#540d6e', '#7f180d', '#00939c',
            '#5f3c99', '#ff4a46', '#8f0075', '#ff3c38', '#83a697',
            '#1e96be', '#d9e021', '#f18d05', '#f6e120', '#8f2d56',
            '#006837', '#e7298a', '#ce1256', '#01665e', '#dfc27d',
            '#35978f', '#bf812d', '#543005', '#8c510a', '#80cdc1',
        ]
        colors_others = 'gray'
        
        figure, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 5))
        ax.patch.set_color("#f5f5f5")
        
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                      left=False, right=False, labelbottom=False, labelleft=False)
        
        plt.grid(color='w', zorder=0, linewidth=2)
        for spine in ax.spines.values():
            spine.set_color('gray')
        
        unique_labels = np.unique(cache_labels)
        for i, label in enumerate(unique_labels):
            mask = cache_labels == label
            if i < len(colors):
                plt.scatter(cache_features_fit[mask, 0], 
                          cache_features_fit[mask, 1],
                          c=colors[i], s=15, marker='x', zorder=5)
            else:
                plt.scatter(cache_features_fit[mask, 0],
                          cache_features_fit[mask, 1],
                          c=colors_others, s=5, zorder=5)
        
        os.makedirs('fig', exist_ok=True)
        save_path = f'fig/cache_features_iter_{iter}.png'
        plt.savefig(save_path)
        plt.close()

def run_test_bpre(pos_cfg, lr_cfg, loader, clip_model, clip_weights, dataset_name, logger):   
    with torch.cuda.amp.autocast():
        pos_cache, accuracies = {}, []
        
        feature_dim = clip_weights.shape[0]
        num_classes = clip_weights.shape[1]

        pos_params = validate_positive_config(pos_cfg)
        pos_enabled = bool(pos_params['enabled'])
        
        prototype_manager = PrototypeManager(
            num_classes=num_classes,
            feature_dim=feature_dim,
            momentum=pos_params['momentum'],
            update_interval=pos_params['update_interval'],
            init_prototypes=clip_weights.T,
        )
        
        reward_calculator = AdaptiveRewardCalculator(
            temperature=pos_params['temperature'],
            warmup_steps=pos_params['warmup_steps'],
            min_reward=pos_params['min_reward'],
            lambda_sim=pos_params['lambda_sim'],
            lambda_conf=pos_params['lambda_conf'],
            lambda_div=pos_params['lambda_div'],
            history_size=pos_params['shot_capacity'],
        )
        
        clip_weights_global = clip_weights.clone()
        num_avg = 0

        for i, (images, target) in enumerate(tqdm(loader, desc='Processing test samples: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local)
            new_clip_weights = text_residue(clip_weights_local)

            # image_features_x, clip_logits, entropy, prob_map, pred = get_clip_logits(
            #     images[0] if dataset_name != 'A' else images, 
            #     clip_model, 
            #     new_clip_weights
            # )
            image_features_x, clip_logits, entropy, prob_map, pred = get_clip_logits(images, clip_model, new_clip_weights)

            target = target.cuda()

            rewards = reward_calculator.compute_reward(
                image_features_x, 
                clip_logits,
                prototype_manager.visual_prototypes
            )

            if pos_enabled:
                entropy = get_entropy(entropy, clip_weights)
                reward_score = float(rewards.detach().mean().item())
                should_cache = entropy <= pos_params['threshold'] or pred not in pos_cache
                if should_cache:
                    update_cache(
                        pos_cache,
                        pred,
                        [image_features_x.detach(), entropy, reward_score],
                        pos_params['shot_capacity']
                    )
                    prototype_manager.update_from_cache(
                        pos_cache,
                        i,
                        pos_params['temperature'],
                    )

                if pos_cache:
                    pos_cache_keys, pos_cache_values, all_classes = cache_key_value(
                        image_features_x,
                        pos_cache,
                        pos_params['alpha'],
                        pos_params['beta'],
                        clip_weights,
                        pos_params['temperature'],
                    )
                    pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
            
            steps = pos_params['steps']
            for _ in range(steps):
                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()
                
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    cache_logits = compute_cache_logits(
                        image_features_x,
                        new_pos_cache_keys,
                        pos_cache_values,
                        pos_params['alpha'],
                        pos_params['beta'],
                        clip_weights
                                        )
                    final_logits += cache_logits
                    final_logits += compute_prototype_logits(
                        image_features_x,
                        prototype_manager.text_prototypes,
                        prototype_manager.visual_prototypes,
                        pos_params['gamma'],
                        pos_params['prototype_logit_scale'],
                    )
                    loss = avg_entropy(final_logits)
                    
                    image2text_loss = InfoNCELoss(
                        new_pos_cache_keys.T,
                        new_clip_weights[:, all_classes].T,
                        pos_params['align_temperature'],
                    )
                    loss += image2text_loss * lr_cfg['align']
                else:
                    loss = avg_entropy(final_logits)
                
                optimizer_params = [
                    {'params': text_residue.parameters(), 'lr': lr_cfg['text']}
                ]
                if pos_enabled and pos_cache:
                    optimizer_params.append(
                        {'params': pos_cache_residue.parameters(), 'lr': lr_cfg['image']}
                    )
                    
                optimizer = torch.optim.AdamW(
                    optimizer_params,
                    eps=pos_params['optimizer_eps'],
                    weight_decay=pos_params['weight_decay']
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images[0] if dataset_name != 'A' else images, 
                    clip_model,
                    new_clip_weights
                )
                
                final_logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(
                        image_features,
                        new_pos_cache_keys,
                        pos_cache_values,
                        pos_params['alpha'],
                        pos_params['beta'],
                        clip_weights
                    )
                    final_logits += compute_prototype_logits(
                        image_features,
                        prototype_manager.text_prototypes,
                        prototype_manager.visual_prototypes,
                        pos_params['gamma'],
                        pos_params['prototype_logit_scale'],
                    )
                
                acc = cls_acc(final_logits, target.cuda())
                accuracies.append(acc)
                
                loss = avg_entropy(final_logits)
                if get_entropy(loss, clip_weights) < pos_params['global_update_threshold']:
                    num_avg += 1
                    clip_weights_global = (
                        clip_weights_global * (num_avg / (num_avg + 1)) +
                        new_clip_weights * (1 / (num_avg + 1))
                    )
            
            if i % pos_params['log_interval'] == 0:
                current_acc = sum(accuracies)/len(accuracies)
                stats = prototype_manager.get_quality_stats()
                logger.info(f"Iteration {i}")
                logger.info(f"Current accuracy: {current_acc:.4f}")
                logger.info(f"Loss: {loss.item():.4f}")

                if pos_enabled and i % pos_params['log_interval'] == 0:
                    visualize_cache(pos_cache, i)

    final_acc = sum(accuracies)/len(accuracies)
    logger.info(f"Final test accuracy: {final_acc:.4f}")
    return final_acc

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True,
                      help='settings of BPRE on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true',
                      help='Accepted for compatibility with existing scripts; BPRE currently logs to local files.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True,
                      help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='/data/zhaozy/qiaoxiaozhen/data/TTA-PT',
                      help='Path to the datasets directory.')
    parser.add_argument('--backbone', dest='backbone', type=str, 
                      choices=['RN50', 'ViT-B/16', 'SigLIP', 'OpenCLIP'], required=True,
                      help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--coop', dest='coop', action='store_true',
                      help='Whether you want to use CoOp weights for initialization.')
    parser.add_argument('--max-samples', dest='max_samples', type=int, default=None,
                      help='Limit evaluation samples for profiling/debugging.')
    return parser.parse_args()

def main():
    args = get_arguments()
    
    if args.backbone in ['RN50', 'ViT-B/16']:
        clip_model, _, preprocess = clip.load(args.backbone)
    elif args.backbone == 'SigLIP':
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-SO400M-14-SigLIP'
        )
        clip_model = clip_model.to('cuda')
    elif args.backbone == 'OpenCLIP':
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
        )
        clip_model = clip_model.to('cuda')
    else:
        raise ValueError(f'Unsupported backbone: {args.backbone}')

    # Process datasets
    results = {}
    for dataset_name in args.datasets.split('/'):
        # Set random seed
        random.seed(1)
        torch.manual_seed(1)
        
        # Setup logger
        logger = setup_logging(dataset_name)
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Load configuration
        cfg = get_config_file(args.config, dataset_name)
        cfg['positive'] = validate_positive_config(cfg.get('positive'))
        logger.info(f"Configuration: {cfg}")
        logger.info(f"CoOp enabled: {args.coop}")
        logger.info(f"Backbone: {args.backbone}")
        
        # Data preparation
        test_loader, classnames, template, cupl_path = build_test_data_loader(
            dataset_name, 
            args.data_root, 
            preprocess
        )
        if args.max_samples is not None:
            subset_size = min(args.max_samples, len(test_loader.dataset))
            subset = torch.utils.data.Subset(test_loader.dataset, range(subset_size))
            test_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=test_loader.batch_size,
                num_workers=8,
                shuffle=True,
                pin_memory=True,
            )

        clip_weights = clip_classifier(
            classnames, 
            template, 
            cupl_path, 
            clip_model, 
            args.coop,
            args.backbone
        )

        # Run BPRE
        acc = run_test_bpre(
            cfg['positive'],
            cfg['learning_rate'],
            test_loader,
            clip_model,
            clip_weights,
            dataset_name,
            logger
        )
        
        results[dataset_name] = acc
        
        logger.info(f"Completed processing dataset {dataset_name}")
        logger.info(f"Final accuracy: {acc:.4f}")

    # Print final results
    logger.info("\nFinal Results Summary:")
    for dataset, acc in results.items():
        logger.info(f"{dataset}: {acc:.4f}")

if __name__ == "__main__":
    main()
