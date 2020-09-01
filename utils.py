import torch
from torch import nn

# based on https://gist.github.com/hermesdt/d9a70dc6b499105afb4be6e0e5cd360f
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
#         self.num_features=num_features
        self.gamma = torch.nn.Parameter(torch.Tensor(num_features))
        self.beta = torch.nn.Parameter(torch.Tensor(num_features))
        self.register_buffer("moving_avg", torch.zeros(num_features))
        self.register_buffer("moving_var", torch.ones(num_features))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("momentum", torch.tensor(momentum))
        self._reset()
    
    def _reset(self):
        self.gamma.data.fill_(1)
        self.beta.data.fill_(0)
    
    def forward(self, x):
#         print(self.num_features, x.size())
        if self.training:
            mean = x.mean(dim=[0,2,3]).detach()
            var = x.var(dim=[0,2,3]).detach()
            self.moving_avg = self.moving_avg * self.momentum + mean.detach() * (1. - self.momentum)
            self.moving_var = self.moving_var * self.momentum + var.detach() * (1. - self.momentum)
        else:
            mean = self.moving_avg.detach()
            var = self.moving_var.detach()
            
        x_norm = (x - mean[None,:,None,None]) / (torch.sqrt(var[None,:,None,None] + self.eps))
        return x_norm * self.gamma[None,:,None,None] + self.beta[None,:,None,None]

###################################################################################################

class NamedStreamAverage:
    def __init__(self):
        self.means = {}
        self.cnt = {}

    def add(self, name, val, cnt, ismean=False):
        if name not in self.means:
            self.means[name] = 0.
            self.cnt[name] = 0
        if ismean:
            val *= cnt
        self.means[name] = (self.means[name] * self.cnt[name] + val) / (self.cnt[name] + cnt)
        self.cnt[name] += cnt

    def reset(self):
        self.means = {}
        self.cnt = {}

    def __getitem__(self, key):
        return self.means[key]

    def __contains__(self, key):
        return key in self.means

    def __iter__(self):
        return iter(self.means.keys())
###################################################################################################
    
# An ordinary implementation of Swish function
# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
class Swish(nn.Module): #MemoryEfficient
    def forward(self, x):
        return SwishImplementation.apply(x)
def swish(x):
    return SwishImplementation.apply(x)

###################################################################################################
import torchvision
import torchvision.transforms as transforms
import torch
WORKER = 0 # 4
def dataset(dataset_name, batch_size):
    if dataset_name == "mnist":
        transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor([0.5]), torch.tensor([0.5])),
        ])

        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            , batch_size=batch_size, shuffle=True, num_workers=WORKER, drop_last=True)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            , batch_size=batch_size, shuffle=False, num_workers=WORKER, drop_last=True)
        img_size = (1,32,32)
    elif dataset_name =="cifar10":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        , batch_size=batch_size, shuffle=True, num_workers=WORKER, drop_last=True)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            , batch_size=batch_size, shuffle=False, num_workers=WORKER, drop_last=True)
        img_size = (3,32,32)
    elif dataset_name =="cifar100":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        , batch_size=batch_size, shuffle=True, num_workers=WORKER, drop_last=True)

        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            , batch_size=batch_size, shuffle=False, num_workers=WORKER, drop_last=True)
        img_size = (3,32,32)
    elif dataset_name =="celeba":
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        
        select_sex = lambda x:x[20] # 1:Man 0:Woman
        # 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young
        
        # link for "img_align_celeba.zip": https://docs.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
        trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.CelebA(root='./data', target_type='attr', split="train", download=True, transform=transform_train, target_transform=select_sex)
        , batch_size=batch_size, shuffle=True, num_workers=WORKER, drop_last=True)

        testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CelebA(root='./data', target_type='attr', split="test", download=True, transform=transform_test, target_transform=select_sex)
        , batch_size=batch_size, shuffle=False, num_workers=WORKER, drop_last=True)
        img_size = (3,32,32)
    else:
        raise
    return trainloader, testloader, img_size

###################################################################################################
class RealNoise():
    def __init__(self, data_loader, d_noise, random_weight=False):
        self.d_noise = d_noise
        self.data_loader = data_loader
        self.random_weight=random_weight
    
    def __iter__(self):
        data_loader_iter = iter(self.data_loader)
        while True:
            try:
                images = [x.view(x.size(0), -1) for (x, _), _ in zip(data_loader_iter, range(self.d_noise))]
                if len(images) != self.d_noise:
                    raise StopIteration
                
                images = torch.stack(images, 1)# batch * d * imagesize
                
                if self.random_weight:
                    w = torch.tensor(np.random.random((x.size(0), self.d_noise)))
                    w /= w.sum(1)[:, None]
                    w *= self.d_noise
                    yield (images*w).sum(1)
                else:
                    yield images.sum(1)
                
            except StopIteration:
                data_loader_iter = iter(self.data_loader)
