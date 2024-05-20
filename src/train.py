import torch
import model
import dataset
import tqdm
import torchvision
import pathlib
import datetime
from torch.utils.tensorboard import SummaryWriter

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
        # print(batch.shape)
        noise_shape = (batch_size, 256, 1, 1)

        generator = generator.to(device).train()
        discriminator = discriminator.to(device).train()
        
        # Discriminator
        real_outputs = discriminator(batch)
        real_label = torch.ones((batch_size, 1)).to(device)
        
        noise_batch = torch.rand(
            noise_shape, dtype=torch.double, device=device)
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs)
        fake_label = torch.zeros((batch_size, 1), device=device)
        
        outputs = torch.cat((real_outputs, fake_outputs), dim=0)
        labels = torch.cat((real_label, fake_label), dim=0)
        
        optimizer_D.zero_grad()
        loss_discriminator = loss_D(outputs, labels, reduction="sum")
        loss_discriminator.backward()
        optimizer_D.step()
        
        # Generator
        noise_batch = torch.rand(
            noise_shape, dtype=torch.double, device=device)
        fake_inputs, low_res_fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs)
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


def test():
    # 画像を生成して、データセットとの分布をはかりたい。
    # 識別器の知らない画像を入力し、正解率をはかりたい
    pass

def train_loop():
     # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_path = 'dataset/Maps_IllustrisTNG_LH_z=0.00'
    train_index_set = list(range(15000))[:10000]
    batch_size = 30
    cmd = dataset.CAMELSMultifieldDataset(
        dir_path=dir_path,
        ids=train_index_set,
        device=device,
    )
    dataloader = torch.utils.data.DataLoader(
        cmd,
        batch_size=batch_size, 
        shuffle=True, 
    )

    # モデルの定義
    generator = model.Generator((batch_size, 256, 1, 1), (3, 256, 256)).double()
    discriminator = model.Discriminator((3, 256, 256))
    loss_D = torchvision.ops.focal_loss.sigmoid_focal_loss
    loss_G = torchvision.ops.focal_loss.sigmoid_focal_loss
    lr = 0.0001
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    output_dir = 'experiment_results'
    
    max_epoch = 2
    
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S') 
    
    with SummaryWriter(f'logs/{time_stamp}') as writer:
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