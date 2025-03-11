from torchvision import transforms

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Chuyển đổi ảnh sang grayscale
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Chuyển đổi ảnh sang grayscale
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return train_transform, test_transform