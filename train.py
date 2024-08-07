import os
import toml
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import torch.utils.data

from trainer import Trainer
from gtcrn import GTCRN
from datasets import MyDataset
from loss_factory import loss_wavmag, loss_mse, loss_hybrid

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.deterministic =True    # 确保每次运行代码时，使用相同的输入和模型参数能够产生相同的输出
# torch.backends.cudnn.benchmark = False  # 防止 cuDNN 在每次输入大小变化时重新选择最佳算法，确保训练过程的稳定性和可重复性


def run(rank, config, args):

    args.rank = rank
    args.device = torch.device(rank)

    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'     # 分布式训练的主节点地址, 单机多卡可以设置为localhost
        os.environ['MASTER_PORT'] = '12354'     # 端口（未使用的）
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()      # 确保所有进程同步

        train_dataset = MyDataset(**config['train_dataset'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                       **config['train_dataloader'], shuffle=False)

        validation_dataset = MyDataset(**config['validation_dataset'])
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                                            **config['validation_dataloader'], shuffle=False)
    else:
        train_dataset = MyDataset(**config['train_dataset'])
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       **config['train_dataloader'], shuffle=True)

        validation_dataset = MyDataset(**config['validation_dataset'])
        validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                            **config['validation_dataloader'], shuffle=False)

    model = GTCRN(**config['network_config'])
    model.to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])

    if config['loss']['loss_func'] == 'wav_mag':
        loss = loss_wavmag()
    elif config['loss']['loss_func'] == 'mse':
        loss = loss_mse()
    elif config['loss']['loss_func'] == 'hybrid':
        loss = loss_hybrid()
    else:
        raise NotImplementedError

    trainer = Trainer(config=config, model=model, optimizer=optimizer, loss_func=loss,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
                      train_sampler=train_sampler, args=args)

    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='cfg_train.toml')
    parser.add_argument('-D', '--device', default='0', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = toml.load(args.config)

    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
