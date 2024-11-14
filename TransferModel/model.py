import torch
import torch.nn as nn
from torch.nn import init
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import time
from concurrent.futures import ThreadPoolExecutor

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    return net

class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return std * mean_variance_norm(content) + mean
    
class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
                                        key_planes=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    
    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
        
        with ThreadPoolExecutor() as executor:
            future_attn_adain_4_1 = executor.submit(
                self.attn_adain_4_1, content4_1, style4_1, content4_1_key, style4_1_key, seed=seed
            )

            future_attn_adain_5_1 = executor.submit(
                self.attn_adain_5_1, content5_1, style5_1, content5_1_key, style5_1_key, seed=seed
            )
            
            attn_adain_4_1_result = future_attn_adain_4_1.result()
            attn_adain_5_1_result = future_attn_adain_5_1.result()

        combined_result = attn_adain_4_1_result + self.upsample5_1(attn_adain_5_1_result)
        return self.merge_conv(self.merge_conv_pad(combined_result))
    
class Decoder(nn.Module):

    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs
    
class AdaAttNModel:
    def __init__(self):
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        image_encoder.load_state_dict(torch.load('TransferModel//pretrained//vgg_normalised.pth', weights_only=True))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.Sequential(*enc_layers[:4]) 
        enc_2 = nn.Sequential(*enc_layers[4:11])
        enc_3 = nn.Sequential(*enc_layers[11:18])
        enc_4 = nn.Sequential(*enc_layers[18:31])
        enc_5 = nn.Sequential(*enc_layers[31:44])

        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # ===========================================================================================

        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder', 'transformer']
        parameters = []
        self.max_sample = 64 * 64

        adaattn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 + 64, max_sample=self.max_sample)
        self.net_adaattn_3 = init_net(adaattn_3)
        self.model_names.append('adaattn_3')
        parameters.append(self.net_adaattn_3.parameters())

        channels = 512 + 256 + 128 + 64
        transformer = Transformer(in_planes=512, key_planes=channels, shallow_layer=True)
        decoder = Decoder(True)
        self.net_decoder = init_net(decoder)
        self.net_transformer = init_net(transformer)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transformer.parameters())

        # ===========================================================================================

        self.net_decoder.load_state_dict(torch.load('TransferModel//pretrained//latest_net_decoder.pth', weights_only=True))
        self.net_transformer.load_state_dict(torch.load('TransferModel//pretrained//latest_net_transformer.pth', weights_only=True))
        self.net_adaattn_3.load_state_dict(torch.load('TransferModel//pretrained//latest_net_adaattn_3.pth', weights_only=True))

        self.c = None
        self.cs = None
        self.s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return mean_variance_norm(feats[last_layer_idx])

    def forward(self, content_img, style_img):

        with ThreadPoolExecutor() as executor:
            future_c_feats = executor.submit(self.encode_with_intermediate, content_img)
            future_s_feats = executor.submit(self.encode_with_intermediate, style_img)

            self.c_feats = future_c_feats.result()
            self.s_feats = future_s_feats.result()

        with ThreadPoolExecutor() as executor:
            future_c_adain_feat_3 = executor.submit(
                self.net_adaattn_3, 
                self.c_feats[2], self.s_feats[2],
                self.get_key(self.c_feats, 2, True), 
                self.get_key(self.s_feats, 2, True),
                self.seed
            )
            
            future_cs = executor.submit(
                self.net_transformer,
                self.c_feats[3], self.s_feats[3],
                self.c_feats[4], self.s_feats[4],
                self.get_key(self.c_feats, 3, True),
                self.get_key(self.s_feats, 3, True),
                self.get_key(self.c_feats, 4, True),
                self.get_key(self.s_feats, 4, True),
                self.seed
            )

            c_adain_feat_3 = future_c_adain_feat_3.result()
            cs = future_cs.result()

        self.cs = self.net_decoder(cs, c_adain_feat_3)
        return self.cs


# def transfer(content_path, style_path, result_path):
#     start = time.time()
#     model = AdaAttNModel()

#     content_img = Image.open(content_path).convert('RGB')
#     style_img = Image.open(style_path).convert('RGB')

#     transformer = transforms.Compose([
#         transforms.Resize((512, 512), interpolation=Image.BICUBIC),
#         transforms.ToTensor()
#     ])
#     # origin_size = (content_img.size[1], content_img.size[0])
#     # resize = transforms.Resize(origin_size)

#     content_img = transformer(content_img)
#     style_img = transformer(style_img)

#     result = model.forward(content_img.unsqueeze(0), style_img.unsqueeze(0))[0]

#     print(f'Finish: {time.time() - start}')

#     save_image(result, result_path)


def transfer(content_img, style_img):
    start = time.time()
    model = AdaAttNModel()

    transformer = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    # origin_size = (content_img.size[1], content_img.size[0])
    # resize = transforms.Resize(origin_size)

    content_img = transformer(content_img)
    style_img = transformer(style_img)

    result = model.forward(content_img.unsqueeze(0), style_img.unsqueeze(0))[0]
    print(f'Finish: {time.time() - start}')
    return to_pil_image(result)
