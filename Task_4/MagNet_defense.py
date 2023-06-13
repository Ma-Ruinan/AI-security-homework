# Attack: 0
# No attack
######################
config['attack_id'] = 0
config['attack_function'] = None
config['epsilons'] = 0

make_dirs(config, test_set)

attack_save_path = os.path.join(save_path, f"Attack-{config['attack_id']}")
for i in tqdm(range(len(test_set)), leave=False, desc=f"Attack: {config['attack_id']}"):
    x, y = test_set[i]
    attack_image_save_path = os.path.join(attack_save_path, str(y), f'{i}.png')
    save_image(x, attack_image_save_path)

# Attack: 1
# AdvGAN
######################
#See subsequent code

# Attack: 2
# FGSM
######################
config['attack_id'] = 2
config['attack_function'] = fb.attacks.FGSM()
config['epsilons'] = 0.1

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)

use_cuda=True
image_nc=1
batch_size = 128

gen_input_nc = image_nc

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_generator_path = './netG_epoch_40.pth'
pretrained_G = Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

mnist_dataset = torchvision.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
for X_mb, Y_mb in enumerate(test_loader, 0):
        X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
        perturbation = pretrained_G(X_mb)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        attack_function = 'AdvGAN'
        adversarial_image = perturbation + X_mb
        image_id = batch_idx * batch_size + i
        y = Y_mb.item()
        attack_image_save_path = os.path.join(attack_save_path, str(y), f'{image_id}.png')
        save_image(adversarial_image, attack_image_save_path)
