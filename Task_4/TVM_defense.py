class TVTransform(object):
    def __init__(self, weight=0.1, max_iter=10):
        self.weight = weight
        self.max_iter = max_iter

    def __call__(self, x):
        x = x.cpu().numpy().transpose(1, 2, 0)
        x = denoise_tv_chambolle(x, weight=self.weight, max_num_iter=self.max_iter)
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        return x

# Define the new transforms with TVM preprocessing
tvm_transform_train = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    TVTransform(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

tvm_transform_test = transforms.Compose([
    transforms.ToTensor(),
    TVTransform(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

tvm_train_set = torchvision.datasets.MNIST(root='./data', split='train', download=True, transform=tvm_transform_train)
tvm_test_set = torchvision.datasets.MNIST(root='./data', split='test', download=True, transform=tvm_transform_test)
tvm_train_loader = torch.utils.data.DataLoader(tvm_train_set, batch_size=64, shuffle=True)
tvm_test_loader = torch.utils.data.DataLoader(tvm_test_set, batch_size=64, shuffle=False)
