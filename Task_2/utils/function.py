import torchvision.datasets as datasets
import torch
import torch.nn as nn


def image_folder_custom_label(root, transform, custom_label):

    old_data = datasets.ImageFolder(root=root, transform=transform)
    # old_classes => ["image_dir/dirname0", "image_dir/dirname1", ...]
    old_classes = old_data.classes

    label2idx = {}
    # idx2label => ["classname0", "classname1", "classname2", ...]
    for i, item in enumerate(custom_label):
        label2idx[item] = i
    # label2idx => ["classname0": 0, "classname1: 1, "classname1": 2, ...]

    new_data = datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=lambda x: custom_label.index(old_classes[x]))
    new_data.classes = custom_label
    new_data.class_to_idx = label2idx

    return new_data


def FGSM(model, image, label, eps, device):

    image = image.to(device)
    image.requires_grad = True
    label = label.to(device)
    model = model.to(device)

    output = model(image)
    model.zero_grad()

    cost = nn.CrossEntropyLoss()
    loss = cost(output, label).to(device)
    loss.backward()

    attack_image = image + eps*image.grad.sign()
    attack_image = torch.clamp(attack_image, min=0, max=1)

    return attack_image


def IFGSM(model, image, label, iteration, step, device):

    image = image.to(device)
    image.requires_grad = True
    image = image.to(device)
    label = label.to(device)
    model = model.to(device)
    cost = nn.CrossEntropyLoss()

    for i in range(iteration):
        output = model(image)
        loss = cost(output, label).to(device)
        grad = torch.autograd.grad(loss, image)[0]
        image = image + step*torch.sign(grad)
        image = torch.clamp(image, min=0, max=1)

    attack_image = image
    return attack_image


def MIFGSM(model, image, label, iteration, eps, device):

    image = image.to(device)
    label = label.to(device)
    model = model.to(device)
    g = torch.zeros_like(image)
    delta = torch.zeros_like(image, requires_grad=True)
    alpha = eps / (iteration / 2)
    decay = 1.0
    cost = nn.CrossEntropyLoss()

    for i in range(iteration):
        output = model(image + delta)
        loss = cost(output, label)

        loss.backward()

        if delta.grad is None:
            continue

        g = decay * g + delta.grad / torch.mean(
            torch.abs(delta.grad), dim=(1, 2, 3), keepdim=True
        )

        delta.data = delta.data + alpha * g.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(image + delta.data, 0, 1) - image

        delta.grad.detach_()
        delta.grad.zero_()

    return image + delta
