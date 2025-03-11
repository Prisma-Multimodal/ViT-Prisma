

from vit_prisma.sae.config import VisionModelSAERunnerConfig
from dataclasses import dataclass

from typing import Literal


@dataclass
class JEPABaseConfig(VisionModelSAERunnerConfig):


    architecture: Literal["standard", "gated", "jumprelu"] = "standard"
    expansion_factor: int = 16

    model_name: str = "vjepa_v1_vit_huge"
    # model_name = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
    checkpoint_path: str = "/checkpoint/firstnamejoseph" # meta cluster
    wandb_project: str = "vjepa_l_sae"

    d_in: int = 1280

    min_l0 = 5
    min_explained_variance = 0.30

    l1_coefficient: float = 0.7
    lr: float = 0.0002

    layer_subtype: str = "hook_mlp_out"

    num_epochs: int = 500

    lr_scheduler_name: str = (
        "cosineannealingwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    )

    lr_warm_up_steps: int = 200
    wandb_log_frequency: int = 100

    cls_token_only: bool = True # Only do CLS token training

    normalize_activations: str = None # What should this be?

    feature_sampling_window: int = 1000
    dead_feature_window: int = 5000
    dead_feature_threshold: float = 1e-08

    device: str = "cuda"

    n_validation_runs: int = 10 # spaced linearly throughout training


    train_batch_size: int = 4096

    use_ghost_grads: bool = False 

    def __post_init__(self):
        # initialize parent
        super().__post_init__()

        if self.cls_token_only:
            self.context_size = 1

        self.hook_point = f"blocks.{self.hook_point_layer}.{self.layer_subtype}" # change hookpoint name here
        self.total_training_images = 1_300_000 * self.num_epochs
        self.total_training_tokens = self.total_training_images * self.context_size 


        print("Running with CLIP Base Config")

#/home/institution/s/firstname.joseph/SAE_factory/src/SAE_factory/quick_train/run_sweep_clip.py --hook_point_layer=9 --l1_coefficient=0.8011371355363572 --lr=0.000292136313548604 --train_batch_size=4096
