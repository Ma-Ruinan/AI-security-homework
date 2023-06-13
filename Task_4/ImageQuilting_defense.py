# Define the patch size
PATCH_SIZE = 96

# Define the number of patches to be used for quilting
NUM_PATCHES = 100

# Define the overlap size between patches
OVERLAP_SIZE = PATCH_SIZE // 6

# Define the input image size
IMAGE_SIZE = 96

# Define the quilting function
def quilt(image, patch_size, num_patches, overlap_size):
    # Convert the image to a numpy array
    image = np.array(image)

    # Compute the dimensions of the image
    height, width, channels = image.shape

    # Compute the number of patches that can be placed horizontally and vertically
    num_horizontal_patches = (width - patch_size) // (patch_size - overlap_size) + 1
    num_vertical_patches = (height - patch_size) // (patch_size - overlap_size) + 1
    # Initialize the output image
    output = np.zeros((height, width, channels))

    # Loop over all the patches
    for i in range(num_vertical_patches):
        for j in range(num_horizontal_patches):
            # Compute the coordinates of the patch
            top = i * (patch_size - overlap_size)
            left = j * (patch_size - overlap_size)
            bottom = top + patch_size
            right = left + patch_size

            # Compute the coordinates of the overlap region
            overlap_top = top + overlap_size
            overlap_left = left + overlap_size
            overlap_bottom = bottom - overlap_size
            overlap_right = right - overlap_size

            # Compute the SSD matrix for the overlap region
            ssd_matrix = np.zeros((num_patches, num_patches))
            for k in range(num_patches):
                for l in range(num_patches):
                    patch1 = image[k * overlap_size:k * overlap_size + overlap_size,
                                    l * overlap_size:l * overlap_size + overlap_size]
                    patch2 = image[k * overlap_size:k * overlap_size + overlap_size,
                                    l * overlap_size:l * overlap_size + overlap_size]
                    ssd_matrix[k, l] = np.sum((patch1 - patch2) ** 2)

            # Compute the optimal seam for the overlap region
            seam = np.zeros((overlap_size, 2))
            seam[:, 0] = overlap_top + np.argmin(ssd_matrix[:, 0])
            seam[:, 1] = overlap_left + np.argmin(ssd_matrix[0, :])
            seam = seam.astype(int)

            # Fill in the overlap region using the optimal seam
            output[overlap_top:overlap_bottom, overlap_left:overlap_right] = \
                image[seam[:, 0], seam[:, 1], :]
            # Convert the output image to a PIL Image object
            output = Image.fromarray(np.uint8(output))

            return output

# Define the transforms for training and testing
transform_train_q = transforms.Compose([
    transforms.Lambda(lambda x: quilt(x, PATCH_SIZE, NUM_PATCHES, OVERLAP_SIZE)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test_q = transforms.Compose([
    transforms.Lambda(lambda x: quilt(x, PATCH_SIZE, NUM_PATCHES, OVERLAP_SIZE)),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_set_q = torchvision.datasets.MNIST(root='./data', split='train', download=True, transform=transform_train_q)
test_set_q = torchvision.datasets.MNIST(root='./data', split='test', download=True, transform=transform_test_q)
train_loader_q = torch.utils.data.DataLoader(train_set_q, batch_size=64, shuffle=True)
test_loader_q = torch.utils.data.DataLoader(test_set_q, batch_size=64, shuffle=False)
