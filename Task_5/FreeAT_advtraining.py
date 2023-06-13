def train(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:]).to(device)
    #mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:]).to(device)
    #std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.to(device)
        target = target.to(device)
        data_time.update(time.time() - end)
        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            #in1.sub_(mean).div_(std)
            model.cuda()            
            output = model(in1)
            if not os.path.isdir(os.path.join('output', "./ffffffff")):
              5	实验方法
代码参考课上给出的：
https://github.com/mahyarnajibi/FreeAdversarialTraining/
下面仅对调整部分以及实验要求相关代码部分展开：
由于我们将要基于MNIST模型展开，所以需要将代码中原有三通道部分全部修改为一通道，以train函数为例，修改后的如下：
def train(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:]).to(device)
    #mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:]).to(device)
    #std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.to(device)
        target = target.to(device)
        data_time.update(time.time() - end)
        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            #in1.sub_(mean).div_(std)
            model.cuda()            
            output = model(in1)
            if not os.path.isdir(os.path.join('output', "./ffffffff")):
                os.makedirs(os.path.join('output', "./ffffffff"))                      
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % configs.TRAIN.print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
                sys.stdout.flush()
                
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
def normaltest():
  model = Net()
  train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
  test_dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
  # Criterion:
  criterion = nn.CrossEntropyLoss().cuda()
    
  # Optimizer:
  optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)
  best = 0
  for epoch in range(10):
      for i, (x, y) in enumerate(train_loader):
          optimizer.zero_grad()
          y_pred = model(x)
          loss = nn.functional.cross_entropy(y_pred, y)
          loss.backward()
          optimizer.step()
          if i % 100 == 0:
              print(f"Epoch {epoch}, Batch {i}, Loss {loss.item():.4f}")
      correct = 0
      total = 0
      with torch.no_grad():
          for x, y in test_loader:
              y_pred = model(x)
              _, predicted = torch.max(y_pred.data, 1)
              total += y.size(0)
              correct += (predicted == y).sum().item()
      test_acc = correct / total
      if(test_acc>best):
        best = test_acc
        torch.save(model.state_dict(), "./model.pth")
      print("Test accuracy:", test_acc)
      print("Best:", best)

  val_loader = torch.utils.data.DataLoader(
        MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
            transforms.Resize(configs.DATA.img_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False)
  for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs)
      
from torchvision.datasets import MNIST
def main():
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value
    
    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    model = Net()
    
    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)
    
    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

            
    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')
    
    train_dataset = MNIST(root="./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True)
    
    normalize = transforms.Normalize(mean=configs.TRAIN.mean,
                                    std=configs.TRAIN.std)
    val_loader = torch.utils.data.DataLoader(
        MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
            transforms.Resize(configs.DATA.img_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        print(' Performing PGD Attacks ')
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs)
        validate(val_loader, model, criterion, configs)
        return
    

    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        adjust_learning_rate(configs.TRAIN.lr, optimizer, epoch, configs.ADV.n_repeats)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join('trained_models', configs.output_name))
        
    # Automatically perform PGD Attacks at the end of training
    print(' Performing PGD Attacks ')
    for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs)
