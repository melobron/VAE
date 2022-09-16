import torch
from torch import nn
import torch.nn.functional as F


############################################## Basic Blocks ##############################################
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ResBlock(nn.Module):
    def __init__(self, n_feats, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        layers = []
        layers += [Conv2dBlock(n_feats, n_feats, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        layers += [Conv2dBlock(n_feats, n_feats, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, n_feats, num_blocks, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        layers = []
        for i in range(num_blocks):
            layers += [ResBlock(n_feats, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


############################################## Encoder/Decoder ##############################################
class Encoder(nn.Module):
    def __init__(self, in_channels, n_feats, n_downsample, n_res, norm, activation, pad_type):
        super(Encoder, self).__init__()

        layers = []
        layers += [Conv2dBlock(in_channels, n_feats, 7, 1, 3, norm=norm, activation=activation, pad_type=pad_type)]

        # Down-sampling Blocks
        for i in range(n_downsample):
            layers += [Conv2dBlock(n_feats, 2 * n_feats, 4, 2, 1, norm=norm, activation=activation, pad_type=pad_type)]
            n_feats *= 2

        # Residual Blocks
        layers += [ResBlocks(n_feats, n_res, norm, activation, pad_type)]

        self.model = nn.Sequential(*layers)
        self.output_dim = n_feats

        # Mean, Logvar
        self.mu = Conv2dBlock(n_feats, n_feats, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)
        self.logvar = Conv2dBlock(n_feats, n_feats, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

    def forward(self, x):
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels, n_feats, n_upsample, n_res, res_norm='adain', activation='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        layers = []

        # Residual Blocks
        layers += [ResBlocks(n_feats, n_res, res_norm, activation, pad_type)]

        # Up-sampling Blocks
        for i in range(n_upsample):
            layers += [nn.Upsample(scale_factor=2),
                       Conv2dBlock(n_feats, n_feats // 2, 5, 1, 2, norm='ln', activation=activation, pad_type=pad_type)]
            n_feats //= 2

        # Reflection padding Conv
        layers += [Conv2dBlock(n_feats, out_channels, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


############################################## VAE ##############################################
class VAE(nn.Module):
    def __init__(self, device, in_channels=1, out_channels=1, n_feats=64, n_downsample=2, n_upsample=2, n_res=6, norm='bn', activation='relu', pad_type='reflect'):
        super(VAE, self).__init__()

        self.device = device

        self.encoder = Encoder(in_channels, n_feats, n_downsample, n_res, norm, activation, pad_type)
        self.decoder = Decoder(out_channels, self.encoder.output_dim, n_upsample, n_res, norm, activation, pad_type)

    def encode(self, images):
        mu, logvar = self.encoder(images)
        latent = torch.randn(mu.size()).to(self.device)
        return mu, logvar, latent

    def decode(self, latent):
        images = self.decoder(latent)
        return images

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


if __name__ == "__main__":
    n_downsample = 2
    multiple = 1
    tile_size = 128 * multiple
    latent_size = 1

    vae = VAE(device='cuda:0', in_channels=1, out_channels=1, n_downsample=n_downsample, n_upsample=n_downsample)

    # img = torch.zeros(size=(3, 1, tile_size, tile_size))
    img = torch.zeros(size=(5, 1, 256, 256))
    mu, var, noise = vae.encode(img)

    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    z = eps * std + mu
    print(z.shape)  # 5, 256, 32, 32 for (2, 2)

    z_channel = 64 * (2**n_downsample)
    z_size = int(256 / (2**n_downsample))
    new_z = torch.zeros(size=(3, z_channel, z_size, z_size))
    recon = vae.decode(new_z)
    print(recon.shape)

