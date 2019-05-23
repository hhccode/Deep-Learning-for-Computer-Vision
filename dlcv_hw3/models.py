import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.utils as vutils
from torch.optim import Adam

class GAN():
    def __init__(self, latent_dim, batch_size, device):
        self.G = Generator(input_dim=latent_dim, filter_num=1024)
        self.D = Discriminator(filter_num=128)
        self.loss_fn = nn.BCELoss()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = device
        self.G_optim = Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_optim = Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.y_real = torch.ones((batch_size, 1), device=device)
        self.y_fake = torch.zeros((batch_size, 1), device=device)

    def move_to_device(self):
        self.G.to(self.device)
        self.D.to(self.device)
        
    def train_D(self, x_real):
        self.D.zero_grad()

        z = torch.randn(x_real.size(0), self.latent_dim, 1, 1, device=self.device)
        x_fake = self.G(z).detach()

        y_real_pred = self.D(x_real)
        real_loss = self.loss_fn(y_real_pred, self.y_real[:x_real.size(0)])
        y_fake_pred = self.D(x_fake)
        fake_loss = self.loss_fn(y_fake_pred, self.y_fake[:x_real.size(0)])
        
        D_loss = real_loss + fake_loss
        D_loss.backward()
        self.D_optim.step()

        return D_loss.cpu().item()

    def train_G(self):
        self.G.zero_grad()

        z = torch.randn(self.batch_size*2, self.latent_dim, 1, 1, device=self.device)
        x_fake = self.G(z)
        y_fake_pred = self.D(x_fake)
        
        # Closer to real is better, which means it can fool the discriminator
        G_loss = self.loss_fn(y_fake_pred, torch.cat((self.y_real, self.y_real)))
        G_loss.backward()
        self.G_optim.step()

        return G_loss.cpu().item()

    def save_D(self, fname):
        torch.save(self.D.state_dict(), "./gan/ckpt/D/{}.pkl".format(fname))
    
    def save_G(self, fname):
        torch.save(self.G.state_dict(), "./gan/ckpt/G/{}.pkl".format(fname))
    
    def save_image(self, fname):
        fig = plt.figure()
        z = torch.load("./model/gan-latent-input.pkl", map_location=self.device)
        
        with torch.no_grad():
            imgs = self.G(z).cpu().detach()

        plt.title("GAN")
        plt.imsave(fname, np.transpose(vutils.make_grid(imgs, padding=2, normalize=True), (1, 2, 0)))
        plt.close("all")

class ACGAN():
    def __init__(self, latent_dim, batch_size, device):
        self.G = Generator(input_dim=latent_dim, filter_num=1024)
        self.D = Discriminator(filter_num=128, acgan=True)
        self.loss_fn = nn.BCELoss()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = device
        self.G_optim = Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_optim = Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.y_real = torch.ones((batch_size, 1), device=device)
        self.y_fake = torch.zeros((batch_size, 1), device=device)

    def load_model(self, G_path, D_path):
        self.G.load_state_dict(torch.load(G_path))
        self.D.load_state_dict(torch.load(D_path))

    def move_to_device(self):
        self.G.to(self.device)
        self.D.to(self.device)
        
    def train_D(self, x_real, real_attr):
        self.D.zero_grad()

        z = torch.randn(x_real.size(0), self.latent_dim, 1, 1, device=self.device)
        fake_attr = torch.randint(0, 2, (x_real.size(0), )).float()
        z[:, -1, 0, 0] = fake_attr
        fake_attr = fake_attr.unsqueeze(1).to(self.device)
        x_fake = self.G(z).detach()

        y_real_dis, y_real_aux = self.D(x_real)
        real_dis_loss = self.loss_fn(y_real_dis, self.y_real[:x_real.size(0)])
        real_aux_loss = self.loss_fn(y_real_aux, real_attr)
        real_loss = real_dis_loss + real_aux_loss

        with torch.no_grad():
            prediction = torch.zeros(x_real.size(0), 1)
            for i in range(x_real.size(0)):
                if y_real_aux[i, 0] > 0.5:
                    prediction[i, 0] = 1.0

            correct = prediction.cpu().eq(real_attr.cpu()).sum()
            acc = float(correct) / x_real.size(0)

        y_fake_dis, y_fake_aux = self.D(x_fake)
        fake_dis_loss = self.loss_fn(y_fake_dis, self.y_fake[:x_real.size(0)])
        fake_aux_loss = self.loss_fn(y_fake_aux, fake_attr)
        fake_loss = fake_dis_loss + fake_aux_loss
        
        D_loss = real_loss + fake_loss
        D_loss.backward()
        self.D_optim.step()

        return D_loss.cpu().item(), acc

    def train_G(self):
        self.G.zero_grad()

        z = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_attr = torch.randint(0, 2, (self.batch_size, )).float()
        z[:, -1, 0, 0] = fake_attr
        fake_attr = fake_attr.unsqueeze(1).to(self.device)
        x_fake = self.G(z)
        y_fake_dis, y_fake_aux = self.D(x_fake)
        
        # Closer to real is better, which means it can fool the discriminator
        G_loss = self.loss_fn(y_fake_dis, self.y_real) + self.loss_fn(y_fake_aux, fake_attr)
        G_loss.backward()
        self.G_optim.step()

        return G_loss.cpu().item()

    def save_D(self, fname):
        torch.save(self.D.state_dict(), "./acgan/ckpt/D/{}.pkl".format(fname))
    
    def save_G(self, fname):
        torch.save(self.G.state_dict(), "./acgan/ckpt/G/{}.pkl".format(fname))
    
    def save_image(self, fname):
        fig = plt.figure()
        z = torch.load("./model/acgan-latent-input.pkl", map_location=self.device)
        
        with torch.no_grad():
            imgs = self.G(z).cpu().detach()

        plt.title("GAN")
        plt.imsave(fname, np.transpose(vutils.make_grid(imgs, nrow=10, padding=2, normalize=True), (1, 2, 0)))
        plt.close("all")

class Generator(nn.Module):
    def __init__(self, input_dim, filter_num=1024):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # k-2p=2
            # 100*1*1 --> 1024*4*4
            nn.ConvTranspose2d(in_channels=input_dim, out_channels=filter_num, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(True),
            # 1024*4*4 --> 512*8*8
            nn.ConvTranspose2d(in_channels=filter_num, out_channels=filter_num//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filter_num//2),
            nn.ReLU(True),
            # 512*8*8 --> 256*16*16
            nn.ConvTranspose2d(in_channels=filter_num//2, out_channels=filter_num//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filter_num//4),
            nn.ReLU(True),
            # 256*16*16 --> 128*32*32
            nn.ConvTranspose2d(in_channels=filter_num//4, out_channels=filter_num//8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filter_num//8),
            nn.ReLU(True),
            # Generator output won't use BN.
            # 128*32*32 --> 3*64*64
            nn.ConvTranspose2d(in_channels=filter_num//8, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.initialize_weights()
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

class Discriminator(nn.Module):
    def __init__(self, filter_num=128, acgan=False):
        super(Discriminator, self).__init__()
        self.acgan = acgan
        self.net = nn.Sequential(
            # Discriminator input won't use BN.
            # Input is 64*64, so kernel uses 4 instead of 3
            # 3*64*64 --> 128*32*32
            nn.Conv2d(in_channels=3, out_channels=filter_num, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 128*32*32 --> 256*16*16
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filter_num*2),
            nn.LeakyReLU(0.2, True),
            # 256*16*16 --> 512*8*8
            nn.Conv2d(in_channels=filter_num*2, out_channels=filter_num*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filter_num*4),
            nn.LeakyReLU(0.2, True),
            # 512*8*8 --> 1024*4*4
            nn.Conv2d(in_channels=filter_num*4, out_channels=filter_num*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=filter_num*8),
            nn.LeakyReLU(0.2, True)
        )
        self.dis = nn.Sequential(
            # 1024*4*4 --> 1*1*1
            nn.Conv2d(in_channels=filter_num*8, out_channels=1, kernel_size=4, bias=False),
            nn.Sigmoid()
        )
        self.aux = nn.Sequential(
            # 1024*4*4 --> 1*1*1
            nn.Conv2d(in_channels=filter_num*8, out_channels=1, kernel_size=4, bias=False),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def forward(self, x):
        x = self.net(x)
        
        dis_x = self.dis(x)
        dis_x = dis_x.view(dis_x.size(0), -1)

        if not self.acgan:
            return dis_x
        else:
            aux_x = self.aux(x)
            aux_x = aux_x.view(aux_x.size(0), -1)

            return dis_x, aux_x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

class CNN(nn.Module):
    def __init__(self, filter_num=64):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # 3*28*28 --> 64*13*13
            nn.Conv2d(in_channels=3, out_channels=filter_num, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25, True),
            # 64*13*13 --> 128*5*5
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num*2, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(filter_num*2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3, True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=filter_num*2*5*5, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamb

        return output, None

def gradient_reverse(x, lamb):
    return GradientReverse.apply(x, lamb)


class DANN(nn.Module):
    def __init__(self, filter_num=64):
        super(DANN, self).__init__()
        self.conv = nn.Sequential(
            # 3*28*28 --> 64*13*13
            nn.Conv2d(in_channels=3, out_channels=filter_num, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25, True),
            # 64*13*13 --> 128*5*5
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num*2, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(filter_num*2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3, True),
        )
        self.class_fc = nn.Sequential(
            nn.Linear(in_features=filter_num*2*5*5, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=10)
        )
        self.domain_fc = nn.Sequential(
            nn.Linear(in_features=filter_num*2*5*5, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, x, lamb):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        reverse_x = gradient_reverse(x, lamb)
        class_output = self.class_fc(x)
        domain_output = self.domain_fc(reverse_x)

        return class_output, domain_output
    
    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()


class GTA():
    def __init__(self, latent_dim, batch_size, device):
        self.F = GTA_F(filter_num=64)
        self.C = GTA_C(channel_num=128)
        self.G = GTA_G(embedding_dim=128, latent_dim=latent_dim, filter_num=512)
        self.D = GTA_D(filter_num=64)
        self.bce_loss_fn = nn.BCELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = device
        self.F_optim = Adam(self.F.parameters(), lr=0.0005, betas=(0.8, 0.999))
        self.C_optim = Adam(self.C.parameters(), lr=0.0005, betas=(0.8, 0.999))
        self.G_optim = Adam(self.G.parameters(), lr=0.0005, betas=(0.8, 0.999))
        self.D_optim = Adam(self.D.parameters(), lr=0.0005, betas=(0.8, 0.999))
        self.y_real = torch.ones((batch_size, 1), device=device)
        self.y_fake = torch.zeros((batch_size, 1), device=device)

    def move_to_device(self):
        self.F.to(self.device)
        self.C.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)
        
    def train_D(self, src_real, src_labels, src_labels_onehot, tgt_real):
        self.D.zero_grad()

        z = torch.randn(src_real.size(0), self.latent_dim, device=self.device)

        src_emb = self.F(src_real)
        self.src_emb = torch.cat((src_emb, src_labels_onehot, z), 1).contiguous().view(src_real.size(0), -1, 1, 1)
        self.src_fake = self.G(self.src_emb)

        src_real_dis, src_real_aux = self.D(src_real)
        src_fake_dis, _ = self.D(self.src_fake)


        z = torch.randn(tgt_real.size(0), self.latent_dim, device=self.device)

        tgt_emb = self.F(tgt_real)
        tgt_emb = torch.cat((tgt_emb, src_labels_onehot, z), 1).contiguous().view(tgt_real.size(0), -1, 1, 1)
        self.tgt_fake = self.G(tgt_emb)

        tgt_fake_dis, _ = self.D(self.tgt_fake)


        src_real_dis_loss = self.bce_loss_fn(src_real_dis, self.y_real[:src_real.size(0)])
        src_real_aux_loss = self.ce_loss_fn(src_real_aux, src_labels)
        src_fake_dis_loss = self.bce_loss_fn(src_fake_dis, self.y_fake[:src_real.size(0)])
        tgt_fake_dis_loss = self.bce_loss_fn(tgt_fake_dis, self.y_fake[:tgt_real.size(0)])

        D_loss = src_real_dis_loss + src_real_aux_loss + src_fake_dis_loss + tgt_fake_dis_loss
        D_loss.backward(retain_graph=True)
        self.D_optim.step()

        return D_loss.cpu().item()

    def train_G(self, src_labels):
        self.G.zero_grad()
        
        src_fake_dis, src_fake_aux = self.D(self.src_fake)

        src_fake_dis_loss = self.bce_loss_fn(src_fake_dis, self.y_real[:src_labels.size(0)])
        src_fake_aux_loss = self.ce_loss_fn(src_fake_aux, src_labels)


        G_loss = src_fake_dis_loss + src_fake_aux_loss
        G_loss.backward(retain_graph=True)
        self.G_optim.step()

        return G_loss.cpu().item()
    
    def train_C(self, src_real, src_labels):
        self.C.zero_grad()

        src_emb = self.F(src_real)
        self.src_classes = self.C(src_emb)

        C_loss = self.ce_loss_fn(self.src_classes, src_labels)
        C_loss.backward(retain_graph=True)
        self.C_optim.step()

        return C_loss.cpu().item()

    def train_F(self, src_labels):
        self.F.zero_grad()

        src_classes_loss = self.ce_loss_fn(self.src_classes, src_labels)

        _, src_fake_aux = self.D(self.src_fake)
        src_fake_aux_loss = self.ce_loss_fn(src_fake_aux, src_labels) * 0.1

        tgt_fake_dis, _ = self.D(self.tgt_fake)
        tgt_fake_dis_loss = self.bce_loss_fn(tgt_fake_dis, self.y_real[:tgt_fake_dis.size(0)]) * 0.1 * 0.3


        F_loss = src_classes_loss + src_fake_aux_loss + tgt_fake_dis_loss
        F_loss.backward()
        self.F_optim.step()

        return F_loss.cpu().item()
    
    def save_C(self, fname):
        torch.save(self.C.state_dict(), fname)
    
    def save_F(self, fname):
        torch.save(self.F.state_dict(), fname)

    def reset_all_weights(self):
        self.F.initialize_weights()
        self.C.initialize_weights()
        self.G.initialize_weights()
        self.D.initialize_weights()

class GTA_F(nn.Module):
    def __init__(self, filter_num=64):
        super(GTA_F, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter_num, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num*2, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.initialize_weights()

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)

        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
 
class GTA_C(nn.Module):
    def __init__(self, channel_num=128):
        super(GTA_C, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=channel_num, out_features=channel_num),
            nn.ReLU(True),
            nn.Linear(in_features=channel_num, out_features=10)
        )

        self.initialize_weights()
    
    def forward(self, x):
        return self.net(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.1)
                nn.init.zeros_(m.bias.data)

class GTA_G(nn.Module):
    def __init__(self, embedding_dim=128, latent_dim=512, filter_num=512):
        super(GTA_G, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embedding_dim+latent_dim+10, out_channels=filter_num, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=filter_num, out_channels=filter_num//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_num//2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=filter_num//2, out_channels=filter_num//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_num//4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=filter_num//4, out_channels=filter_num//8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_num//8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=filter_num//8, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        self.initialize_weights()

    def forward(self, x):
        return self.net(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.zeros_(m.bias.data)

class GTA_D(nn.Module):
    def __init__(self, filter_num=64):
        super(GTA_D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_num*2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=filter_num*2, out_channels=filter_num*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_num*4),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=filter_num*4, out_channels=filter_num*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter_num*2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(4, 4),
        )
        self.dis = nn.Sequential(
            nn.Linear(in_features=filter_num*2, out_features=1),
            nn.Sigmoid()
        )
        self.aux = nn.Sequential(
            nn.Linear(in_features=filter_num*2, out_features=10)
        )

        self.initialize_weights()

    def forward(self, x):
        x = self.net(x).squeeze()

        dis_x = self.dis(x)
        dis_x = dis_x.view(dis_x.size(0), -1)
        aux_x = self.aux(x)
        aux_x = aux_x.view(aux_x.size(0), -1)

        return dis_x, aux_x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.1)
                nn.init.zeros_(m.bias.data)

class ADDA():
    def __init__(self, batch_size, device):
        self.src_encoder = ADDA_Encoder()
        self.tgt_encoder = ADDA_Encoder()
        self.classifier = ADDA_Classifier()
        self.discriminator = ADDA_Discriminator()
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.tgt_optim = Adam(self.tgt_encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.discri_optim = Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.y_src = torch.ones((batch_size,), device=device)
        self.y_tgt = torch.zeros((batch_size,), device=device)

    def pretrain_src(self, src_loader):
        optimizer = Adam(list(self.src_encoder.parameters())+list(self.classifier.parameters()), lr=0.0001, betas=(0.5, 0.999))

        self.src_encoder.to(self.device)
        self.classifier.to(self.device)
        self.src_encoder.train()
        self.classifier.train()

        EPOCH = 10

        for epoch in range(EPOCH):

            LOSS = 0.0
            ACC = 0.0

            for i, data in enumerate(src_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.classifier(self.src_encoder(inputs))
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    predict = torch.max(outputs, 1)[1]
                    acc = np.mean((predict == labels).cpu().numpy())
                    LOSS += loss.item()
                    ACC += acc
                
                print("\rEpoch:[{}/{}], Step:[{}/{}], Loss:{:.5f}, Acc:{:.5f}".format(epoch+1, EPOCH, i, len(src_loader), loss.item(), acc), end="")
            
            print("\n\n\nEpoch {}, Loss = {:.5f}, Acc = {:.5f}".format(epoch+1, LOSS/len(src_loader), ACC/len(src_loader)))
            
            torch.save(self.src_encoder.state_dict(), "./mnistm-svhn/src_encoder/epoch{}-acc{:.5f}.pkl".format(epoch+1, ACC/len(src_loader)))
            torch.save(self.classifier.state_dict(), "./mnistm-svhn/classifier/epoch{}-acc{:.5f}.pkl".format(epoch+1, ACC/len(src_loader)))
            
    def load_pretrain(self, encoder_path, classifier_path):
        self.src_encoder.load_state_dict(torch.load(encoder_path))
        self.classifier.load_state_dict(torch.load(classifier_path)) 
    
    def train_discri(self, src, tgt):
        self.discri_optim.zero_grad()

        src_feat = self.src_encoder(src)
        tgt_feat = self.tgt_encoder(tgt)

        feat = torch.cat((src_feat, tgt_feat), 0).detach()
        label = torch.cat((self.y_src, self.y_tgt)).long()

        outputs = self.discriminator(feat)
        loss = self.loss_fn(outputs, label)
        loss.backward()

        self.discri_optim.step()

        with torch.no_grad():
            predict = torch.max(outputs, 1)[1]
            acc = np.mean((predict == label).cpu().numpy())
        
        return loss.cpu().item(), acc
    
    def train_tgt(self, tgt):
        self.tgt_optim.zero_grad()
        self.discri_optim.zero_grad()

        tgt_feat = self.tgt_encoder(tgt)
        label = self.y_src.long()

        outputs = self.discriminator(tgt_feat)
        loss = self.loss_fn(outputs, label)
        loss.backward()
        
        self.tgt_optim.step()

        return loss.cpu().item()
    
    def save_tgt_encoder(self, fname):
        torch.save(self.tgt_encoder.state_dict(), fname)

class ADDA_Encoder(nn.Module):
    def __init__(self):
        super(ADDA_Encoder, self).__init__()
        self.net = nn.Sequential(
            # 3*28*28 --> 64*13*13
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25, True),
            # 64*13*13 --> 128*5*5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3, True),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.contiguous().view(x.size(0), -1)
        return x

class ADDA_Classifier(nn.Module):
    def __init__(self):
        super(ADDA_Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=128*5*5, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ADDA_Discriminator(nn.Module):
    def __init__(self):
        super(ADDA_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=128*5*5, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=2),
        )

    def forward(self, x):
        x = self.net(x)
        return x