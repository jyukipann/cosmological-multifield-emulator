import os
import pathlib
import datetime
import tqdm
import torch
import torchvision.transforms.functional
import torchvision
from torcheval.metrics import BinaryAccuracy
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator
import dataset
import utils
import loss

import numpy as np
# import warnings
# warnings.resetwarnings()
# warnings.simplefilter('error')

def train(
        dataloader, 
        generator, discriminator, 
        optimizer_G, optimizer_D, 
        epoch, device, noise_weight_rate, summary_writer:SummaryWriter=None):
    
    max_step = len(dataloader)
    loss_sum_D = 0
    loss_sum_G = 0
    rec_loss_func = loss.ReconstructionLoss()
    focal_loss_func = loss.FocalLoss()

    for i, (batch, params) in tqdm.tqdm(enumerate(dataloader), desc=f"train epoch {epoch}", total=max_step):
        batch_size = batch.shape[0]

        # batchにノイズを付加する
        batch = (1 - noise_weight_rate)*batch + noise_weight_rate*torch.randn_like(batch)

        batch:torch.tensor = batch.to(device)
        params:torch.tensor = params.to(device)
        _, params_dim = params.size()
        params = torch.reshape(params, (-1, params_dim, 1, 1))

        # 画像を縮小して、loss計算に使う
        batch_low_res =  torchvision.transforms.functional.resize(
            batch, (128, 128))
        # print(batch.shape)
        noise_shape = (batch_size, 256, 1, 1)

        generator: Generator = generator.to(device).train()
        discriminator: Discriminator = discriminator.to(device).train()
        
        # Discriminator
        real_outputs = discriminator(batch, batch_low_res)
        real_outputs, real_low_res_maps = real_outputs[0], real_outputs[1:]

        # Discriminator学習のための正解の生成
        real_label = torch.ones((batch_size, 1)).to(device)
        # Label Smoothing
        real_label *=  0.9
        
        # 画像生成用のノイズ生成 正規分布
        noise_batch = torch.randn(noise_shape).to(device)
        # paramsを生成用ノイズに混入
        noise_batch[:, :params_dim] = params
        # 生成
        fake_inputs, low_res_fake_inputs = generator(noise_batch)

        # fake_inputsにノイズを付加
        fake_inputs = (1 - noise_weight_rate)*fake_inputs + noise_weight_rate*torch.randn_like(fake_inputs)

        fake_outputs = discriminator(fake_inputs, low_res_fake_inputs)[0]

        # Discriminator学習のための正解の生成
        fake_label = torch.ones((batch_size, 1), device=device)
        # Label Smoothing
        fake_label *= 0.1
        
        # バッチに混ぜるのは良くないとも聞くのでバラでbackward()を二回するのはあり
        # outputs = torch.cat((real_outputs, fake_outputs), dim=-1)
        # labels = torch.cat((real_label, fake_label), dim=-1)

        # Loss計算とパラメータ更新
        optimizer_D.zero_grad()
        # もとのロス        
        rec_loss = 0
        rec_loss += rec_loss_func(real_low_res_maps[0], batch_low_res)
        rec_loss += rec_loss_func(real_low_res_maps[1], batch_low_res)
        rec_loss += rec_loss_func(real_low_res_maps[2], batch_low_res)
        rec_loss /= 3
        
        # focal_loss = focal_loss_func(real_outputs, real_label)
        # focal_loss += focal_loss_func(fake_outputs, fake_label)
        # focal_loss /= 2

        # loss_discriminator = focal_loss + rec_loss


        # Wasserstain loss
        # 上記のloss計算とこのloss計算はどっちかだけ使う
        # gradient_penalty = 0.1  # 仮
        gradient_penalty = calculate_gradient_penalty(discriminator, batch.data, fake_inputs.data)
        lambda_gp = 10          # 仮
        
        lossD_r = torch.mean(torch.minimum(0, -1 + real_outputs))  # real loss
        lossD_f = torch.mean(torch.minimum(0, -1 - fake_outputs))  # fake loss
        loss_discriminator = -lossD_r - lossD_f + rec_loss
        loss_discriminator += gradient_penalty * lambda_gp
        loss_discriminator.backward()
        optimizer_D.step()
        
        # Generator
        # 画像生成用のノイズ生成 正規分布
        noise_batch = torch.randn(noise_shape).to(device)
        # paramsを生成用ノイズに混入
        noise_batch[:, :params_dim] = params
        # 生成
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs, low_res_fake_inputs)[0]
        fake_label = torch.ones((batch_size, 1), device=device)
        
        # Loss計算とパラメータ更新
        optimizer_G.zero_grad()
        loss_generator = focal_loss_func(fake_outputs, fake_label)
        loss_generator.backward()
        optimizer_G.step()
        
        loss_sum_D += loss_discriminator
        loss_sum_G += loss_generator
        if summary_writer is not None:
            global_step = int(((epoch-1)+(i/max_step))*1000)
            summary_writer.add_scalar(
                "train/Loss_G", loss_generator, global_step)
            # summary_writer.add_scalar(
            #     "train/Loss_D_reconstruction", rec_loss, global_step)
            # summary_writer.add_scalar(
            #     "train/Loss_D_focal", focal_loss, global_step)
            summary_writer.add_scalar(
                "train/Loss_D", loss_discriminator, global_step)
    print(f"{epoch=} : loss_generator={loss_sum_G/max_step}, loss_discriminator={loss_sum_D/max_step}")


def val(
        dataloader, 
        generator, discriminator, 
        epoch, device, 
        threshold=0.5, summary_writer:SummaryWriter=None):
    max_step = len(dataloader)
    metric = BinaryAccuracy(threshold=threshold)
    accuracy_sum = 0
    for i, (batch, params) in tqdm.tqdm(enumerate(dataloader), desc=f"val epoch {epoch}", total=max_step):
        batch_size = batch.shape[0]
        batch = batch.to(device)
        params = params.to(device)
        noise_shape = (batch_size, 256, 1, 1)
        _, params_dim = params.size()
        params = torch.reshape(params, (-1, params_dim, 1, 1))

        batch_low_res =  torchvision.transforms.functional.resize(
            batch, (128, 128))
        
        generator:Generator = generator.to(device).eval()
        discriminator:Discriminator = discriminator.to(device).eval()
        
        # ノイズ生成
        noise_batch = torch.rand(noise_shape).to(device)
        # paramsを生成用ノイズに混入
        noise_batch[:, :params_dim] = params

        # 生成
        fake_high_res, fake_low_res = generator(noise_batch)
        
        # 推論
        fake_results = discriminator(fake_high_res, fake_low_res)[0]
        real_results = discriminator(batch, batch_low_res)[0]
        
        # 正解生成
        target0 = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        target1 = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
        # 結果の結合
        results = torch.cat([fake_results, real_results], dim=0).flatten()
        target = torch.cat([target0, target1], dim=0).flatten()
        
        # 正解率計算
        metric.update(results, target)
        accuracy = metric.compute()
        accuracy_sum += accuracy
        
    accuracy = accuracy_sum / max_step
    if summary_writer is not None:
        summary_writer.add_scalar("val/D_accuracy", accuracy, epoch)
        
        # 生成画像を可視化
        mgas,hi,b = fake_high_res[0].cpu().detach().numpy()
        # 生成画像をreverse pipelineで表示できる形式に戻す
        mgas = utils.DATA_NORMALIZE_PIPELINE_REVERSE['Mgas'](mgas)
        hi = utils.DATA_NORMALIZE_PIPELINE_REVERSE['HI'](hi)
        b = utils.DATA_NORMALIZE_PIPELINE_REVERSE['B'](b)
        # RGB画像に変換
        mgas = utils.plot_map(mgas, utils.PREFIX_CMAP_DICT['Mgas'])
        hi = utils.plot_map(hi, utils.PREFIX_CMAP_DICT['HI'])
        b = utils.plot_map(b, utils.PREFIX_CMAP_DICT['B'])
        # 横に並べて一枚に結合
        mgas_hi_b = torch.cat([mgas, hi, b], dim=2)
        summary_writer.add_image("val/mgas_hi_b", mgas_hi_b, epoch)
    print(f"{accuracy=}")

def noise_schedule(epoch, coffitent, half_epoch, cutoff_epoch=None)->float:
    if cutoff_epoch >= epoch:
        return 1/(1+torch.e**(coffitent*(epoch-half_epoch)))
    return 0

def calculate_gradient_penalty(D, real_img, fake_img):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_img.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_img + ((1 - alpha) * fake_img)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.Tensor(real_img.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_loop():
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device=}')
    
    # dataset pathoutput_dir/f
    dir_path = 'dataset/normalization/Maps_IllustrisTNG_LH_z=0.00'
    
    # output path   
    output_dir = f'experiment_results/{time_stamp}'
    log_dir = f'experiment_results/{time_stamp}'

    # split dataset
    index_set = list(range(15000))
    train_index_set = index_set[:10000]
    val_index_set = index_set[10000:11500]
    test_index_set = index_set[11500:15000]

    # train configs
    batch_size = 100
    max_epoch = 300
    val_inerval = 10
    checkpoint_interval = 10
    lr = 0.0001


    # import torchvision.transforms as transforms
    # image_size = (256,256)
    # transform = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.CenterCrop(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # dataset_train = torchvision.datasets.CelebA(
    #     root='dataset/CelebA', split='train', transform=transform, download=True)
    # dataset_valid = torchvision.datasets.CelebA(
    #     root='dataset/CelebA', split='valid', transform=transform, download=True)
    # dataset_test = torchvision.datasets.CelebA(
    #     root='dataset/CelebA', split='test',  transform=transform, download=True)
    
        # split dataset
    index_set = list(range(15000))
    train_index_set = index_set[:10000]
    val_index_set = index_set[10000:11500]
    test_index_set = index_set[11500:15000]

    # データセットとデータローダーのインスタンス化
    cmd_train = dataset.CAMELSMultifieldDataset(
        dir_path=dir_path,
        ids=train_index_set,
    )
    
    dataloader_train = torch.utils.data.DataLoader(
        cmd_train,
        # dataset_train,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    cmd_val = dataset.CAMELSMultifieldDataset(
        dir_path=dir_path,
        ids=val_index_set,
    )
    dataloader_val = torch.utils.data.DataLoader(
        cmd_val,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )

    # Model
    generator = Generator((256, 1, 1), (3, 256, 256))
    discriminator = Discriminator((3, 256, 256))

    # Optimizer
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # log dir
    output_dir = pathlib.Path(output_dir)
    log_dir = pathlib.Path(log_dir)
    if 'IS_GPGPU' in os.environ and bool(os.environ['IS_GPGPU']):
        output_dir = '..' / output_dir
        log_dir = '..' / log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    max_noise = 0.8 # 50%
    noise_rate = max_noise / (max_epoch-1)

    # open SummaryWriter
    print(f"Log Dir {log_dir}")
    with SummaryWriter(log_dir) as writer:
        # Train loop
        for epoch in range(1, max_epoch+1):
            # discriminatorに渡す画像の重みの計算
            # noise_weight_rate = max_noise - (noise_rate*(epoch - 1))
            noise_weight_rate = noise_schedule(epoch, 0.03, 100, 200)
            train( 
                dataloader_train, 
                generator, discriminator, 
                optimizer_G, optimizer_D, 
                epoch, device, noise_weight_rate, 
                summary_writer=writer)
            # val interval
            if epoch % val_inerval == 0:
                val(dataloader_val, 
                    generator, discriminator, 
                    epoch, device,
                    threshold=0.5,
                    summary_writer=writer)
            # save checkpoint
            if epoch % checkpoint_interval == 0:
                # save models
                g_name = str(output_dir/f'generator_checkpoint_{epoch}epoch.pth')
                d_name = str(output_dir/f'discriminator_checkpoint_{epoch}epoch.pth')
                torch.save(generator, g_name)
                torch.save(discriminator, d_name)
                print(f"model saved: {g_name}")
                print(f"model saved: {d_name}")

    # save models
    g_name = str(output_dir/f'generator.pth')
    d_name = str(output_dir/f'discriminator.pth')
    torch.save(generator, g_name)
    torch.save(discriminator, d_name)
    print(f"model saved: {g_name}")
    print(f"model saved: {d_name}")


  


def main():
    train_loop()

if __name__ == '__main__':
   main()