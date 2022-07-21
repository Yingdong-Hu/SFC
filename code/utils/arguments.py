import argparse
import torch
import random
from pathlib import Path
import os


def test_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Label Propagation')

    # Datasets
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--manualSeed', type=int, default=777, help='manual seed')

    parser.add_argument('--batchSize', default=1, type=int,
                        help='batchSize')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='temperature')
    parser.add_argument('--topk', default=10, type=int,
                        help='k for kNN')
    parser.add_argument('--radius', default=12, type=float,
                        help='spatial radius to consider neighbors from')
    parser.add_argument('--videoLen', default=20, type=int,
                        help='number of context frames')

    parser.add_argument('--cropSize', default=320, type=int,
                        help='resizing of test image, -1 for native size')

    parser.add_argument('--filelist', default='/scratch/ajabri/data/davis/val2017.txt', type=str)
    parser.add_argument('--save-path', default='./results', type=str)

    # Model Details
    parser.add_argument('--semantic-model', default=None, type=str)
    parser.add_argument('--fc-model', default=None, type=str)
    parser.add_argument('--lambd', default=1.75, type=float,)

    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')
    parser.add_argument('--finetune', default=0, type=int, help='')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Set seed
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def train_args():
    parser = argparse.ArgumentParser(description='Fine-grained Correspondence Training')
    parser.add_argument('--batch-size', default=96, type=int, help='Per GPU batch size.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of total epochs to run.')

    # Model parameters
    parser.add_argument('--arch', type=str, choices=['resnet18', 'resnet50'], default='resnet18',
                        help='Name of architecture to train.')
    parser.add_argument('--projection-hidden-dim', type=int, default=2048, help='Projector hidden dimension.')
    parser.add_argument('--projection-dim', type=int, default=256, help='Projector output dimension.')
    parser.add_argument('--prediction-hidden-dim', type=int, default=2048, help='Predictor hidden dimension.')
    parser.add_argument('--momentum-target', type=float, default=0.99,
                        help='''Base EMA parameter for target network update.
                        The value is increased to 1 during training with cosine schedule.''')
    parser.add_argument('--pos-radius', type=float, default=0.5, help='Positive radius.')
    parser.add_argument('--remove-stride-layers', type=str, nargs='+', default=('layer3', 'layer4'),
                        help='Reduce the stride of some layers in order to obtain a higher resolution feature map.')

    # Optimization parameters
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay')

    # Augmentation parameters
    parser.add_argument('--img-size', default=256, type=int, help='Image input size.')
    parser.add_argument('--min-crop', type=float, default=0.0, help='Minimum scale for random cropping.')
    parser.add_argument('--max-crop', type=float, default=1.0, help='Maximum scale for random cropping.')

    # Dataset parameters
    parser.add_argument('--data-path', default='', help='Dataset path.')
    parser.add_argument('--frame-rate', default=30, type=int, help='Frame rate (fps) of dataset.')
    parser.add_argument('--clip-len', default=1, type=int, help='Number of frames per clip.')
    parser.add_argument('--clips-per-video', default=80, type=int, help='Maximum number of clips per video to consider.')
    parser.add_argument('--workers', default=8, type=int)

    parser.add_argument('--output-dir', default='../output/fc_pretrained', help='Path where to save.')
    parser.add_argument('--save-ckpt-freq', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use for training.')
    parser.add_argument('--seed', type=int, default=777, help='Manual seed.')

    # Weights and Biases arguments
    parser.add_argument('--enable-wandb', default=False,
                        help="Enable logging to Weights and Biases.")
    parser.add_argument('--project', default='SFC', type=str,
                        help="The name of the W&B project where you're sending the new run.")

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args