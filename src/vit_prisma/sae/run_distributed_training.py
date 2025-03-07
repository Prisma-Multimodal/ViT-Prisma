from vit_prisma.sae.training.distributed import *
from vit_prisma.sae.config import VisionModelSAERunnerConfig
from vit_prisma.sae.train_sae import VisionSAETrainer
import argparse, os
from vit_prisma.models.model_loader import load_hooked_model
from torchvision.datasets import ImageFolder
from torchvision import transforms

import logging
import wandb

def setup_logging(rank):
    """Configure logging to print only on rank 0"""
    # Configure the root logger
    root_logger = logging.getLogger()
    
    if rank == 0:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)  # Or logging.ERROR to suppress more messages
    
    # Modify all existing loggers
    for name in logging.root.manager.loggerDict:
        if rank != 0:
            logging.getLogger(name).setLevel(logging.ERROR)

def setup_wandb(args, rank):
    """Initialize wandb only on rank 0"""
    if rank == 0:
        # Only rank 0 should log to wandb
        run_id = args.run_id if hasattr(args, 'run_id') else None
        run = wandb.init(
            project="jepa_vit_large_patch16_sae",
            entity="perceptual-alignment",
            id=run_id,
            resume="allow" if run_id else None,
            config=args.__dict__
        )
        return run
    else:
        # Turn off wandb for other ranks
        os.environ["WANDB_MODE"] = "disabled"
        return None

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed SAE Training')
    
    # Distributed training parameters
    parser.add_argument('--dist-url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', help='distributed backend')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0, 
                        help='GPU id within this node (automatically set by torch.distributed.launch)')

    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training')
    
    # Model and training parameters
    parser.add_argument('--model-name', type=str, required=False, help='Model name', default= 'vjepa_v1_vit_large_patch16')
    # parser.add_argument('--dataset-path', type=str, required=False, help='Path to dataset', default = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC')
    # parser.add_argument('--dataset-name', type=str, default='imagenet1k', help='Dataset name')
    # parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    # parser.add_argument('--config-path', type=str, help='Path to config file')
    
    args = parser.parse_args()

    if args.local_rank is not None:
        args.rank = args.local_rank
    
    # Initialize distributed environment
    init_distributed_mode(args)

    # Initialize distributed training    
    # Set up logging and printing
    setup_logging(args.rank)
    setup_for_distributed(args.rank == 0)
    
    # Initialize wandb only on rank 0
    run = setup_wandb(args, args.rank)

    sae_cfg_path = '/home/mila/s/sonia.joseph/ViT-Prisma/src/vit_prisma/jepa_development/jepa_l_config.json'
    cfg = VisionModelSAERunnerConfig().load_config(sae_cfg_path)

    print(cfg)

    
    # Add distributed parameters to config
    cfg.distributed = args.gpu is not None
    cfg.rank = args.rank
    cfg.world_size = args.world_size
    cfg.gpu = args.gpu
    cfg.dist_url = args.dist_url
    
    # Load model
    model_path = '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/vitl16.pth.tar'
    model = load_hooked_model(args.model_name, local_path = model_path, pretrained=False)

    if model.cfg.video_num_frames > 1:
        def forward_prehook(module, input):
            input = input[0]  # [B, C, H, W]
            input = input.unsqueeze(2).repeat(1, 1, model.cfg.video_num_frames, 1, 1)
            return (input)
        model.register_forward_pre_hook(forward_prehook)
        
    # Load dataset
    normalization = ((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))

   
    # Setup data
    resolution = 224
    transform = transforms.Compose([
                transforms.Resize(size=int(resolution * 256/224)),
                transforms.CenterCrop(size=resolution),
                transforms.ToTensor(),
                transforms.Normalize(normalization[0], normalization[1])])

    train_data_path =  "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/train"
    val_data_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets/ILSVRC/Data/CLS-LOC/val"
    train_dataset = ImageFolder(root=train_data_path, transform=transform)
    val_dataset = ImageFolder(root=val_data_path, transform=transform)


    # Create and run trainer
    trainer = VisionSAETrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    sae = trainer.run()
    
    print(f"Training completed. Rank {cfg.rank} finished.")