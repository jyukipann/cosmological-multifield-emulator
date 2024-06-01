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
    for i, (batch, params) in tqdm.tqdm(enumerate(dataloader), desc=f"train epoch {epoch}", total=max_step):
        batch_size = batch.shape[0]
        batch:torch.tensor = batch.to(device)
        params:torch.tensor = params.to(device)
        _, params_dim = params.size()
        params = torch.reshape(params, (-1, params_dim, 1, 1))
        batch_low_res =  torchvision.transforms.functional.resize(batch, (128, 128))
        # print(batch.shape)
        noise_shape = (batch_size, 256, 1, 1)

        generator:Generator = generator.to(device).train()
        discriminator:Discriminator = discriminator.to(device).train()
        
        # Discriminator
        real_outputs = discriminator(batch, batch_low_res)
        real_label = torch.ones((batch_size, 1)).to(device)
        
        noise_batch = torch.rand(noise_shape).to(device)
        # paramsを生成用ノイズに混入
        noise_batch[:, :params_dim] = params
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs, low_res_fake_inputs)
        fake_label = torch.zeros((batch_size, 1), device=device)
        
        outputs = torch.cat((real_outputs, fake_outputs), dim=-1)
        labels = torch.cat((real_label, fake_label), dim=-1)

        optimizer_D.zero_grad()
        loss_discriminator = loss_D(outputs, labels, reduction="sum")
        loss_discriminator.backward()
        optimizer_D.step()
        
        # Generator
        noise_batch = torch.rand(noise_shape).to(device)
        # paramsを生成用ノイズに混入
        noise_batch[:, :params_dim] = params
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs, low_res_fake_inputs)
        fake_label = torch.ones((batch_size, 1), device=device)
        optimizer_G.zero_grad()
        loss_generator = loss_G(fake_outputs, fake_label)
        loss_generator.backward()
        optimizer_G.step()
        
        loss_sum_D += loss_discriminator
        loss_sum_G += loss_generator
        if summary_writer is not None:
            global_step = int(((epoch-1)+(i/max_step))*1000)
            summary_writer.add_scalar(
                "train/Loss_G", loss_generator, global_step)
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
        batch_low_res =  torchvision.transforms.functional.resize(batch, (128, 128))
        # print(batch.shape)
        noise_shape = (batch_size, 256, 1, 1)

        generator:Generator = generator.to(device).eval()
        discriminator:Discriminator = discriminator.to(device).eval()
        noise_batch = torch.rand(noise_shape).to(device)
        fake_high_res, fake_low_res = generator(noise_batch)
        
        fake_results = discriminator(fake_high_res, fake_low_res)
        real_results = discriminator(batch, batch_low_res)
        target0 = torch.zeros((batch_size, 1), device=device)
        target1 = torch.ones((batch_size, 1), device=device)
        results = torch.cat([fake_results, real_results], dim=0).flatten()
        target = torch.cat([target0, target1], dim=0).flatten()
        # print(f"{target0.size()=} {target1.size()=} {fake_results.size()=} {real_results.size()=}")
        # print(f"{target.size()=}")
        # print(f"{results.size()=}")
        metric.update(results, target)
        accuracy = metric.compute()
        accuracy_sum += accuracy
        
    accuracy = accuracy_sum / max_step
    if summary_writer is not None:
        summary_writer.add_scalar("val/D_accuracy", accuracy, epoch)
        
        # 生成画像を可視化
        mgas,hi,b = fake_high_res[0].cpu().detach().numpy()
        mgas = utils.plot_map(mgas, utils.PREFIX_CMAP_DICT['Mgas'])
        hi = utils.plot_map(hi, utils.PREFIX_CMAP_DICT['HI'])
        b = utils.plot_map(b, utils.PREFIX_CMAP_DICT['B'])
        mgas_hi_b = torch.cat([mgas, hi, b], dim=2)
        summary_writer.add_image("val/mgas_hi_b", mgas_hi_b, epoch)
    print(f"{accuracy=}")

def train_loop():
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Use {device=}')
    
    # dataset path
    dir_path = 'dataset/Maps_IllustrisTNG_LH_z=0.00'
    
    # output path   
    output_dir = f'experiment_results/{time_stamp}'
    log_dir = f'experiment_results/{time_stamp}'

    # split dataset
    index_set = list(range(15000))
    train_index_set = index_set[:10000]
    val_index_set = index_set[10000:11500]
    test_index_set = index_set[11500:15000]

    # train configs
    batch_size = 50
    max_epoch = 300
    val_inerval = 1
    checkpoint_interval = 10
    lr = 0.0001

    # データセットとデータローダーのインスタンス化
    cmd_train = dataset.CAMELSMultifieldDataset(
        dir_path=dir_path,
        ids=train_index_set,
    )
    dataloader_train = torch.utils.data.DataLoader(
        cmd_train,
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

    # Loss
    loss_D = torchvision.ops.focal_loss.sigmoid_focal_loss
    loss_G = torch.nn.HuberLoss()

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

    # open SummaryWriter
    print(f"Log Dir {log_dir}")
    with SummaryWriter(log_dir) as writer:
        # Train loop
        for epoch in range(1, max_epoch+1):
            train(
                dataloader_train, 
                generator, discriminator, 
                loss_G, loss_D, 
                optimizer_G, optimizer_D, 
                epoch, device,
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