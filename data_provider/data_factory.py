import torch
from data_provider.data_loader import Dataset_Basketball
from data_provider.transforms import RandomErasing
from torch.utils.data import DataLoader
from torchvision import transforms as Transforms

# 自定义 collate_fn 解决 tuple 问题
def custom_collate_fn_train(batch):
    # 分别提取原始图片和增强图片
    person_image_norm_bs = torch.stack([item[1] for item in batch])  # 获取原始图片
    person_image_augment_bs = torch.stack([item[0] for item in batch])  # 获取增强图片

    # 拼接 (2*B, C, H, W)
    # combined_images = torch.cat((person_image_norm_bs, person_image_augment_bs), dim=0)
    
    return person_image_augment_bs

# 自定义 collate_fn 解决 tuple 问题
def custom_collate_fn_val(batch):
    # 分别提取原始图片和增强图片
    person_image_norm_bs = torch.stack([item[1] for item in batch])  # 获取原始图片
    person_image_augment_bs = torch.stack([item[0] for item in batch])  # 获取增强图片

    # 拼接 (2*B, C, H, W)
    # combined_images = torch.cat((person_image_norm_bs, person_image_augment_bs), dim=0)
    
    return person_image_norm_bs


def data_provider(args, stage):
    # 加载目标检测模型
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # person_detect_model_weights = "/homec/xiaolei/projects/ultralytics/weights/yolov9e.pt"
    # person_detect_model = YOLO(person_detect_model_weights)
    # person_detect_model = person_detect_model.to(device)
    
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size

    transformer_train = Transforms.Compose([
        Transforms.ToTensor(),
        Transforms.Resize((args.height, args.width)),
        Transforms.RandomHorizontalFlip(p=0.5),
        Transforms.Pad(10),
        Transforms.RandomCrop((args.height, args.width)),
        RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformer_val = Transforms.Compose([
        Transforms.Resize((args.height, args.width)),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if stage == 'train':
        train_data_set = Dataset_Basketball(
            root_dir = args.root_dir_train, 
            transform_train = transformer_train,
            transform_val = transformer_val
        )
        train_data_loader = DataLoader(
            train_data_set,
            batch_size=batch_size,
            collate_fn=custom_collate_fn_train,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return train_data_loader
    else:
        test_data_set = Dataset_Basketball(
            root_dir = args.root_dir_valid, 
            transform_train = transformer_train,
            transform_val = transformer_val
        )
        test_data_loader = DataLoader(
            test_data_set,
            batch_size=batch_size,
            collate_fn=custom_collate_fn_val,
            shuffle=False,
            num_workers=0,
            drop_last=False)
        return test_data_loader

