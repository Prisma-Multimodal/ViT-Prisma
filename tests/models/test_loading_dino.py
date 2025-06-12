import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms

import pytest
from tqdm import tqdm

from vit_prisma.dataloaders.imagenet_dataset import load_imagenet
from vit_prisma.models.base_vit import HookedViT
from transformers import ViTModel

@pytest.mark.skip(reason="Requires pretrained weights")
def test_loading_dino():
    TOLERANCE = 1e-4

    model_name = "facebook/dino-vitb16"
    batch_size = 5
    channels = 3
    height = 224
    width = 224
    device = "cpu"

    dino_model = ViTModel.from_pretrained(model_name)

    hooked_model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=False, fold_ln=False)
    hooked_model.to(device)

    with torch.random.fork_rng():
        torch.manual_seed(1)
        input_image = torch.rand((batch_size, channels, height, width)).to(device)
    with torch.no_grad():
        dino_output, hooked_output = dino_model(input_image), hooked_model(input_image)

    cls_token = dino_output.last_hidden_state[:, 0]
    patches = dino_output.last_hidden_state[:, 1:]
    patches_pooled = patches.mean(dim=1)
    dino_processed_output = torch.cat((cls_token.unsqueeze(-1), patches_pooled.unsqueeze(-1)), dim=-1)
    assert torch.allclose(hooked_output, dino_processed_output, atol=TOLERANCE), f"Model output diverges! Max diff: {torch.max(torch.abs(hooked_output - dino_processed_output))}"

@pytest.mark.skip(reason="This is a utility function")
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]

class LinearClassifier(nn.Module):
    """A simple linear layer on top of frozen features."""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

@pytest.mark.skip(reason="This is a utility function")
def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    if url is not None:
        print("Loading pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )["state_dict"]
        new_state_dict = {
            'linear.weight': state_dict['module.linear.weight'],
            'linear.bias': state_dict['module.linear.bias']
        }
        linear_classifier.load_state_dict(new_state_dict, strict=True)
    else:
        print("Using random linear weights.")


@torch.no_grad()
@pytest.mark.skip(reason="This is a utility function")
def validate_network(val_loader, model, linear_classifier):
    model.eval()
    linear_classifier.eval()
    total_loss = 0.0
    total_samples = 0
    total_acc1 = 0.0
    total_acc5 = 0.0
    criterion = nn.CrossEntropyLoss()

    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")
    for inp, target in progress_bar:
        inp = inp.to('cuda')
        target = target.to('cuda')
        features = model(inp)
        output = linear_classifier(features)
        loss = criterion(output, target)
        batch_size = inp.size(0)
        total_loss += loss.item() * batch_size
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        total_acc1 += acc1.item() * batch_size
        total_acc5 += acc5.item() * batch_size
        total_samples += batch_size
        progress_bar.set_postfix({
            'Acc@1': f"{acc1.item():.2f}",
            'Acc@5': f"{acc5.item():.2f}",
            'Loss': f"{loss.item():.2f}"
        })

    avg_loss = total_loss / total_samples
    avg_acc1 = total_acc1 / total_samples
    avg_acc5 = total_acc5 / total_samples

    print(f"*Avg. Val Acc@1: {avg_acc1:.3f} Avg. Val Acc@5: {avg_acc5:.3f} Avg. Val Loss: {avg_loss:.3f}")
    return {"loss": avg_loss, "acc1": avg_acc1, "acc5": avg_acc5}


@pytest.mark.skip(reason="Requires external datasets and pretrained weights")
def test_accuracy_baseline_hooked_model():
    model_name = 'facebook/dino-vitb16'
    hooked_model = HookedViT.from_pretrained(model_name)
    hooked_model = hooked_model.to('cuda')
    hooked_model.eval()

    linear_classifier = LinearClassifier(1536, num_labels=1000).cuda()
    load_pretrained_linear_weights(linear_classifier, 'vit', 16)
    print("Classifier and model loaded.")

    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_path = "/network/scratch/s/sonia.joseph/datasets/kaggle_datasets"
    imagenet_val = load_imagenet(
        preprocess_transform=val_transform,
        dataset_path=dataset_path,
        dataset_type='imagenet1k-val'
    )

    val_loader = torch.utils.data.DataLoader(
        imagenet_val,
        batch_size=512,
        pin_memory=True,
    )

    test_stats = validate_network(val_loader, hooked_model, linear_classifier)
    print(f"Accuracy on {len(imagenet_val)} test images: {test_stats['acc1']:.1f}%")
    # I get [Avg. Val Acc@1: 76.890, Avg. Val Acc@5: 93.228] on Hooked Model; benchmarked by Meta in the dino repo is [Avg. Val Acc@1: 78.162, Avg. Val Acc@5: 93.924]
