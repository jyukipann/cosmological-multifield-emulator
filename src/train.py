import torch
import model
import dataset
import tqdm
import torchvision
import pathlib
import datetime

def train(
        dataloader, 
        generator, discriminator, 
        loss_G, loss_D, 
        optimizer_G, optimizer_D, 
        epoch, device):
    
    for batch, params in tqdm.tqdm(dataloader, desc=f"epoch {epoch}"):
        batch_size = batch.shape[0]
        batch = batch.to(device)
        
        generator = generator.to(device).train()
        discriminator = discriminator.to(device).train()
        
        # Discriminator
        real_outputs = discriminator(batch)
        real_label = torch.ones((batch_size, 1)).to(device)
        
        noise_batch = torch.rand(
            (batch_size, 1, 2, 2), 
            dtype=torch.float64, 
            device=device)
        fake_inputs = generator(noise_batch)
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
            (batch_size, 1, 2, 2), 
            dtype=torch.float64, 
            device=device)
        fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs)
        fake_label = torch.ones((batch_size, 1), device=device)
        optimizer_G.zero_grad()
        loss_generator = loss_G(fake_outputs, fake_label, reduction="sum")
        loss_generator.backward()
        optimizer_G.step()


def test():
    # 画像を生成して、データセットとの分布をはかりたい。
    # 識別器の知らない画像を入力し、正解率をはかりたい
    pass

def train_loop():
     # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_paths = [
        [
            'dataset/Maps_Mgas_IllustrisTNG_CV_z=0.00.npy',
            'dataset/Maps_HI_IllustrisTNG_CV_z=0.00.npy',
            'dataset/Maps_B_IllustrisTNG_CV_z=0.00.npy',
        ],
    ]
    params_paths = ['dataset/params_CV_IllustrisTNG.txt',]
    
    cmd = dataset.CAMELSMultifieldDataset(
        data_paths=data_paths,
        params_paths=params_paths,
        device=device,
    )
    dataloader = torch.utils.data.DataLoader(
        cmd,
        batch_size=10, 
        shuffle=True, 
    )

    # モデルの定義
    generator = model.Generator((1,2,2), (3, 256, 256))
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
    for epoch in range(max_epoch):
        train(
            dataloader, 
            generator, discriminator, 
            loss_G, loss_D, 
            optimizer_G, optimizer_D, 
            epoch, device)
        
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # save models
    torch.save(generator, str(output_dir/f'generator_{time_stamp}.pth'))
    torch.save(discriminator, str(output_dir/f'discriminator_{time_stamp}.pth'))

if __name__ == '__main__':
   train_loop()