import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


from jepa.models.attentive_pooler import AttentiveClassifier


from collections import OrderedDict

# partial
from functools import partial

from vit_prisma.dataloaders.imagenet_dataset import load_imagenet
from vit_prisma.models.model_loader import load_hooked_model
from jepa.models.vision_transformer import VisionTransformer

import torch.nn.functional as F

from torchvision.datasets import ImageFolder

# # train sae
from vit_prisma.sae.train_sae import VisionSAETrainer
from vit_prisma.sae.config import VisionModelSAERunnerConfig


DEVICE = 'cuda'


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_sdpa=True, **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=20, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_sdpa=True, **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_gigantic(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1664, depth=48, num_heads=16, mpl_ratio=64/13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

import torch

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy for the top-k predictions and returns the predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        topk_preds = output.topk(maxk, dim=1, largest=True, sorted=True).indices

        # Compute top-1 accuracy
        top1_acc = 100.0 * topk_preds[:, 0].eq(target).sum().float() / batch_size

        # Compute top-k accuracy
        correct = topk_preds.eq(target.view(-1, 1))  # Shape: (batch_size, maxk)
        accs = [100.0 * correct[:, :k].float().sum() / batch_size for k in topk]

        return accs, topk_preds[:, 0], topk_preds[:, :5]




class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value"""
        self.val = float(val)  # Ensure val is a float
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0  

def evaluate_imagenet_probe(encoder, val_loader, probe, device='cuda', use_bfloat16=True):

    
    top1 = AverageMeter()
    top5 = AverageMeter()

    MAX = 100
    count = 0
    
   

    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(val_loader)):

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

                images = images.to(DEVICE)
                target = target.to(DEVICE)

            

                if i == 0:
                    for j in range(16):
                        img_path = val_loader.dataset.samples[j][0]  # Get image path
                        true_label = target[j].item()
                        print(f"  {img_path} → Label: {true_label}")
                    # break
                                    

                # Get representations
                features = encoder(images)
                # features = features.to(torch.float32)
                # features = features.mean(dim=1).unsqueeze(1)

             


                # # normalize along 1 and second dimension
                # features = F.normalize(features, p=2, dim=1)

                # # try random features
                # features = torch.randn(16, 1568, 1024).to(DEVICE)



                output = probe(features)

                # Measure accuracy
                (acc1, acc5), top1_preds, top5_preds = accuracy(output, target, topk=(1, 5))
                print("\n🔥 True Labels:", target.tolist())
                print("🔥 Top-1 Predictions:", top1_preds.tolist())
                print("🔥 Top-5 Predictions:", top5_preds.tolist())


            
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

            del features, output

            count += 1
            if count > MAX:
                break
        
    
    return top1.avg, top5.avg

def load_attentive_probe(encoder, probe_path):
        # Setup probe


    checkpoint = torch.load(probe_path, map_location=DEVICE)
    checkpoint = checkpoint['classifier']


    # If the checkpoint was saved with DataParallel/DDP (has 'module.' prefix)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace('module.', '')  # remove 'module.' prefix if it exists
        new_state_dict[name] = v

    try:
        embed_dim = encoder.embed_dim
    except:
        embed_dim = encoder.cfg.d_model
    
    try:
        num_heads = encoder.num_heads
    except:
        num_heads = encoder.cfg.n_heads



    classifier = AttentiveClassifier(embed_dim= embed_dim ,num_heads = num_heads, depth=1, num_classes=1000).to(DEVICE)


    classifier.load_state_dict(new_state_dict, strict=True)
    # print(f"Missing keys: {missing}", f"Unexpected keys: {unexpected}")

    # freez params
    # for param in classifier.parameters():
    #     param.requires_grad = False

    # # Put in eval mode for inference
    # classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    # for module in classifier.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.LayerNorm):
    #         module.eval()

    classifier.train()
    
    classifier.to(DEVICE)
    return classifier

def load_model(model_name, path):

    # Load and setup models
    model = torch.load(path)

    if 'vit_large' in model_name:
        encoder = vit_large(tubelet_size=2, num_frames=16)
    elif 'vit_huge' in model_name:
        encoder = vit_huge(tubelet_size=2, num_frames=16)
    # encoder = VisionTransformer(
    #     patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, uniform_power=True,
    #     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), tubelet_size=2, num_frames=16, use_sdpa=True,
    # )

    # Load weights
    print(model.keys())
    encoder_dict = model['target_encoder']
    new_state_dict = {k.replace('module.backbone.', ''): v for k, v in encoder_dict.items()}
    encoder.load_state_dict(new_state_dict)

    if encoder.num_frames > 1:
        def forward_prehook(module, input):
            input = input[0]  # [B, C, H, W]
            input = input.unsqueeze(2).repeat(1, 1, encoder.num_frames, 1, 1)
            return (input)
    encoder.register_forward_pre_hook(forward_prehook)

    # freeze model
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(DEVICE)
    encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    for module in encoder.modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.LayerNorm):
            module.eval()
    encoder.eval()
    return encoder





normalization = ((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
# Setup data
resolution = 224
transform = transforms.Compose([
            transforms.Resize(size=int(resolution * 256/224)),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1])])

classifier_model_library = { # model_name: (model_path, probe_path)
    'vjepa_v1_vit_large_patch16': ('/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/vitl16.pth.tar', '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/probes/in1k-probe.pth.tar.1'),
    'vjepa_v1_vit_huge_patch16_224': ('/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-h-16/vith16.pth.tar', '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-h-16/probes/in1k-probe.pth.tar')

}

sae_cfg = VisionModelSAERunnerConfig().load_config('jepa_l_config.json')


model_name = 'vjepa_v1_vit_large_patch16'

model_path, probe_path = classifier_model_library[model_name]
encoder = load_hooked_model(model_name, local_path = model_path, pretrained=False)

if encoder.cfg.video_num_frames > 1:
    def forward_prehook(module, input):
        input = input[0]  # [B, C, H, W]
        input = input.unsqueeze(2).repeat(1, 1, encoder.cfg.video_num_frames, 1, 1)
        return (input)
encoder.register_forward_pre_hook(forward_prehook)

encoder = encoder.to(DEVICE)



classifier = load_attentive_probe(encoder, probe_path)

data = {}
train_data_path =  "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
val_data_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"
train_data = ImageFolder(root=train_data_path, transform=transform)
val_data = ImageFolder(root=val_data_path, transform=transform)


trainer = VisionSAETrainer(sae_cfg, encoder, train_data, val_data)

trainer.run()

# val_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

# top1_acc, top5_acc = evaluate_imagenet_probe(encoder, val_loader, classifier)
# print(f'Top-1 Accuracy: {top1_acc:.2f}')
# print(f'Top-5 Accuracy: {top5_acc:.2f}')