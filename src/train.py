import torch
import torch.optim as optim
import Generator
import Discriminator

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの読み込み
# ...

# モデルの定義
generator = Generator(z_dim, img_dim).to(device)
discriminator = Discriminator(img_dim).to(device)

# オプティマイザの定義
g_optimizer = optim.Adam(generator.parameters())
d_optimizer = optim.Adam(discriminator.parameters())

# 損失関数の定義
BCE_loss = torch.nn.BCELoss()

# 学習ループ
for epoch in range(num_epochs):
  for i, data in enumerate(dataloader):
    # 訓練データとラベルの準備
    real_img = data[0].to(device)
    label = real_img.size(0) * torch.ones(1).to(device)

    # 生成された画像の生成
    z = torch.randn(label.size()).to(device)
    fake_img = generator(z)

    # Discriminatorの更新
    d_optimizer.zero_grad()
    real_output = discriminator(real_img)
    fake_output = discriminator(fake_img.detach())
    real_loss = BCE_loss(real_output, label)
    fake_loss = BCE_loss(fake_output, 1 - label)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    # Generatorの更新
    g_optimizer.zero_grad()
    fake_output = discriminator(fake_img)
    g_loss = BCE_loss(fake_output, label)
    g_loss.backward()
    g_optimizer.step()
