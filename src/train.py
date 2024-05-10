import torch
import model
import dataset
import tqdm
import torchvision

def train(dataloader, generator, discriminator, loss, optim_D, optim_G, epoch, device):
    for batch in tqdm.tqdm(dataloader):
        batch_size = batch.shape[0]
        batch = batch.to(device)
        
        real_outputs = discriminator(batch)
        real_label = torch.ones((batch_size, 1)).to(device)
        
        noise_batch = torch.rand(
            (batch_size, 1, 2, 2), 
            dtype=torch.float64, 
            device=device)
        fake_inputs = generator(noise_batch)
        fake_outputs = discriminator(fake_inputs)
        fake_label = torch.zeros((batch_size, 1)).to(device)
        
        outputs = torch.cat((real_outputs, fake_outputs), dim=0)
        labels = torch.cat((real_label, fake_label), dim=0)
        
        loss_D = loss(outputs, labels)
        

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
    )
    dataloader = torch.utils.data.DataLoader(
        cmd,
        batch_size=10, 
        shuffle=True, 
    )

    # モデルの定義
    generator = model.Generator((1,2,2), (3, 256, 256)).to(device)
    discriminator = model.Discriminator((3, 256, 256)).to(device)
    
    loss_D = torchvision.ops.focal_loss
    
    max_epoch = 10
    for epoch in range(max_epoch):
        train(dataloader, generator, discriminator, epoch)

if __name__ == '__main__':
   train_loop()