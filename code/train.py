import datetime
import os
import time
import random
from typing import Iterable

import torch
import torch.utils.data
from torchvision import transforms

from data.video import YoutubeVOS
from models import resnet
from models.fine_grained import FineGrained
import utils
from utils.clip_sampler import RandomClipSamplerFrame
from utils import transform_coord


total_step = 0


def main(args):
    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)

    # ============ preparing data ============
    # simple augmentation
    transform_1 = transform_coord.Compose([
        transform_coord.RandomResizedCropCoord(args.img_size, scale=(args.min_crop, args.max_crop)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    transform_2 = transform_1
    transform_train = (transform_1, transform_2)

    dataset = YoutubeVOS(
        args.data_path,
        frames_per_clip=args.clip_len,
        frame_rate=args.frame_rate,
        transform=transform_train)

    train_sampler = RandomClipSamplerFrame(dataset.resampling_idxs, args.clips_per_video)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True)
    args.num_steps = len(data_loader)
    print("Number of video frames = %d" % len(dataset))
    print('Number of training steps per epoch = %d' % len(data_loader))

    if args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    # ============ building fine-grained correspondence network ============
    encoder = resnet.__dict__[args.arch]
    model = FineGrained(encoder, args)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, data_loader,
                        optimizer, device, epoch, args,
                        wandb_logger=wandb_logger)
        if args.output_dir and (epoch + 1 == args.epochs or epoch % args.save_ckpt_freq == 0):
            save_model(args, epoch, model, optimizer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    args=None, wandb_logger=None):

    global total_step
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for step, (videos, coords) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_step += 1
        start_time = time.time()

        video1 = videos[0].to(device, non_blocking=True)
        video2 = videos[1].to(device, non_blocking=True)
        coord1 = coords[0].to(device, non_blocking=True)
        coord2 = coords[1].to(device, non_blocking=True)
        loss = model(video1, video2, coord1, coord2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(video1.shape[0] / (time.time() - start_time))

        if wandb_logger is not None and total_step % 100 == 0:
            wandb_logger.log(dict(loss=loss.item()))
            wandb_logger.log(dict(learning_rate=optimizer.param_groups[0]['lr']))


def save_model(args, epoch, model, optimizer):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args
    }
    torch.save(to_save, os.path.join(args.output_dir, 'pretrained_fc.pth'))


if __name__ == "__main__":
    args = utils.arguments.train_args()
    main(args)