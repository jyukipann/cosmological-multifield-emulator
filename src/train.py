import os
import pathlib
import datetime
import tqdm
import torch
import torchvision.transforms.functional
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
import dataset
# import warnings
# warnings.resetwarnings()
# warnings.simplefilter('error')

def train(
        dataloader, 
        generator, discriminator, 
        loss_G, loss_D, 
        optimizer_G, optimizer_D, 
        epoch, device, summary_writer:SummaryWriter=None):
    
    max_step = len(dataloader)
    loss_sum_D = 0
    loss_sum_G = 0
    for i, (batch, params) in tqdm.tqdm(enumerate(dataloader), desc=f"epoch {epoch}", total=max_step):
        batch_size = batch.shape[0]
        batch = batch.to(device)
        batch_low_res =  torchvision.transforms.functional.resize(batch, (128, 128))
        # print(batch.shape)
        noise_shape = (batch_size, 256, 1, 1)

        generator:Generator = generator.to(device).train()
        discriminator:Discriminator = discriminator.to(device).train()
        
        # Discriminator
        real_outputs = discriminator(batch, batch_low_res)
        real_label = torch.ones((batch_size, 1)).to(device)
        
        noise_batch = torch.rand(noise_shape).to(device)
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs, low_res_fake_inputs)
        fake_label = torch.zeros((batch_size, 1), device=device)
        
        outputs = torch.cat((real_outputs, fake_outputs), dim=0)
        labels = torch.cat((real_label, fake_label), dim=0)
        
        optimizer_D.zero_grad()
        loss_discriminator = loss_D(outputs, labels, reduction="sum")
        loss_discriminator.backward()
        optimizer_D.step()
        
        # Generator
        noise_batch = torch.rand(noise_shape).to(device)
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs, low_res_fake_inputs)
        fake_label = torch.ones((batch_size, 1), device=device)
        optimizer_G.zero_grad()
        loss_generator = loss_G(fake_outputs, fake_label, reduction="sum")
        loss_generator.backward()
        optimizer_G.step()
        
        loss_sum_D += loss_discriminator
        loss_sum_G += loss_generator
        if summary_writer is not None:
            summary_writer.add_scalar(
                "Loss/train/Loss_G", loss_generator, (epoch-1)+(i/max_step))
            summary_writer.add_scalar(
                "Loss/train/Loss_D", loss_discriminator, (epoch-1)+(i/max_step))
    print(f"{epoch=} : loss_generator={loss_sum_G/max_step}, loss_discriminator={loss_sum_D/max_step}")


def val(
        dataloader, 
        generator, discriminator, 
        epoch, device, 
        threshold=0.5, summary_writer:SummaryWriter=None):
    # 画像を生成して、データセットとの分布をはかりたい。
    # 識別器の知らない画像を入力し、正解率をはかりたい
    max_step = len(dataloader)
    for i, (batch, params) in tqdm.tqdm(enumerate(dataloader), desc=f"epoch {epoch}", total=max_step):
        batch_size = batch.shape[0]
        batch = batch.to(device)
        batch_low_res =  torchvision.transforms.functional.resize(batch, (128, 128))
        # print(batch.shape)
        noise_shape = (batch_size, 256, 1, 1)

        generator:Generator = generator.to(device).eval()
        discriminator:Discriminator = discriminator.to(device).eval()
        noise_batch = torch.rand(noise_shape).to(device)
        fake_high_res, fake_low_res = generator(noise_batch)
        
        fake_results = discriminator(fake_high_res, fake_low_res)
        accuracy =  torcheval
        
        
        if summary_writer is not None:
            summary_writer

def train_loop():
     # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device=}')
    dir_path = 'dataset/Maps_IllustrisTNG_LH_z=0.00'
    index_set = list(range(15000))
    train_index_set = index_set[:10000]
    val_index_set = index_set[10000:11500]
    test_index_set = index_set[11500:15000]
    batch_size = 100
    cmd = dataset.CAMELSMultifieldDataset(
        dir_path=dir_path,
        ids=train_index_set,
    )
    dataloader = torch.utils.data.DataLoader(
        cmd,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )

    # モデルの定義
    generator = Generator((256, 1, 1), (3, 256, 256))
    discriminator = Discriminator((3, 256, 256))
    loss_D = torchvision.ops.focal_loss.sigmoid_focal_loss
    loss_G = torchvision.ops.focal_loss.sigmoid_focal_loss
    lr = 0.0001
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    max_epoch = 2
    output_dir = 'experiment_results'
    output_dir = pathlib.Path(output_dir)
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = f'logs/{time_stamp}'
    log_dir = pathlib.Path(log_dir)
    if 'IS_GPGPU' in os.environ and bool(os.environ['IS_GPGPU']):
        output_dir = '..' / output_dir
        log_dir = '..' / log_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir) as writer:
        for epoch in range(max_epoch):
            train(
                dataloader, 
                generator, discriminator, 
                loss_G, loss_D, 
                optimizer_G, optimizer_D, 
                epoch, device)

    # save models
    torch.save(generator, str(output_dir/f'generator_{time_stamp}.pth'))
    torch.save(discriminator, str(output_dir/f'discriminator_{time_stamp}.pth'))

if __name__ == '__main__':
   train_loop()