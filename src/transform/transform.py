import PIL.Image as Image
from torchvision import transforms
from .augment import ImgAugTransform

data_transforms1 = {
    'train':transforms.Compose([
        transforms.CenterCrop(32),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90, 90)),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270, 270)),
            ])
        ]),
        # ImgAugTransform(),
        # lambda x : Image.fromarray(x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

}

data_transforms2 = {
    'train':transforms.Compose([
        transforms.CenterCrop(48),
        # ImgAugTransform(),
        # lambda x: Image.fromarray(x),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90, 90)),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270, 270)),
            ])
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_transforms3 = {
    'train':transforms.Compose([
        # ImgAugTransform(),
        # lambda x : Image.fromarray(x),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90, 90)),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270, 270)),
            ])
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
