import os
import json
import argparse
import torch
from torchvision import transforms
from torchvision import models
from torchvision.utils import save_image
import torch.utils.data as DataLoader
from utils.function import image_folder_custom_label
from utils.function import FGSM, IFGSM, MIFGSM

parser = argparse.ArgumentParser()
parser.add_argument("--eps", default=10/255)
parser.add_argument("--step", default=1/255)
parser.add_argument("--iteration", default=10)
parser.add_argument("--cuda_id", default="0")
parser.add_argument("--image_dir", default="./image")
parser.add_argument("--class_index", default="./imagenet_class_index.json")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_idx = json.load(open(args.class_index))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

normal_data = image_folder_custom_label(root=args.image_dir,
                                        transform=transform,
                                        custom_label=idx2label)
normal_loader = DataLoader.DataLoader(normal_data, batch_size=1, shuffle=False)

model = models.resnet18(pretrained=True).to(device)
model.eval()

for images, labels in normal_loader:
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
    # Clean
    _, pre_ori = torch.max(output.data, dim=1)
    print(f"True label: {pre_ori.item()} | name: {idx2label[pre_ori]}")
    # FGSM
    fgsm_img = FGSM(model=model,
                    image=images,
                    label=labels,
                    eps=args.eps,
                    device=device)
    save_image(fgsm_img, "./adv_img/fgsm.png")

    fgsm_out = model(fgsm_img)

    _, pre_fgsm = torch.max(fgsm_out.data, dim=1)
    print(f"FGSM label: {pre_fgsm.item()} | name: {idx2label[pre_fgsm]}")
    # IFGSM
    ifgsm_img = IFGSM(model=model,
                      image=images,
                      label=labels,
                      iteration=10,
                      step=args.step,
                      device=device)
    save_image(ifgsm_img, "./adv_img/ifgsm.png")

    ifgsm_out = model(ifgsm_img)

    _, pre_ifgsm = torch.max(fgsm_out.data, dim=1)
    print(f"IFGSM label: {pre_ifgsm.item()} | name: {idx2label[pre_ifgsm]}")
    # MIFGSM
    mifgsm_img = MIFGSM(model=model,
                        image=images,
                        label=labels,
                        iteration=10,
                        eps=args.eps,
                        device=device)
    save_image(mifgsm_img, "./adv_img/mifgsm.png")

    mifgsm_out = model(mifgsm_img)

    _, pre_mifgsm = torch.max(mifgsm_out.data, dim=1)
    print(f"MIFGSM label: {pre_mifgsm.item()} | name: {idx2label[pre_mifgsm]}")
