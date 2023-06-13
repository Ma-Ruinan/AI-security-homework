from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import random

def jpeg_preprocessing_defense(image):

    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=75)
    buffer.seek(0)
    jpeg_image = Image.open(buffer).convert('RGB')
    
    processed_image = jpeg_image.filter(ImageFilter.SMOOTH_MORE)

    processed_image = processed_image.resize((96, 96))
    
    tensor_image = transforms.ToTensor()(processed_image)
    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalized_image = normalize(tensor_image)

    return normalized_image

class RandomJPEGPreprocessing:
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            preprocessed_image = jpeg_preprocessing_defense(image)
        else:
            preprocessed_image = transforms.ToTensor()(image)

        return preprocessed_image

transform_train = transforms.Compose([
    RandomJPEGPreprocessing(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_set = torchvision.datasets.MNIST(root='./data', split='train', download=True, transform=transform_train)
test_set = torchvision.datasets.MNIST(root='./data', split='test', download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
