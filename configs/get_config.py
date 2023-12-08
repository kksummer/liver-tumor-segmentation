import yaml
from utils.mean_std_computing import compute_mean_std
from dataset.dataset import LiverDataset
from datetime import datetime
import torch
import json
import os
import importlib
import torch.optim as optim
from utils.early_stopping import EarlyStopping
from dataset.transform import Preprocess


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()
        self.dataset_config = self.config['dataset']
        self.model_config = self.config['model']
        self.runtime_config = self.config['runtime']
        self.runtime = type("Runtime", (), self.config['runtime'])

    def load_config(self):
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

    def save_info(self, log_info_list=None, mode=None):
        dataset_type = self.dataset_config['type']
        model_type = self.model_config['type']
        progress = self.runtime_config['progress']

        # 保存 json 文件
        env = self.runtime_config['env']
        log_info = ("{}-{}-{}".format(model_type, progress, env))
        json_save_path = os.path.join("logs", dataset_type, "{}.json".format(log_info))

        # logs/LiTs/UNet-liver_training-ian1.json
        if mode == 'start':
            if os.path.exists(json_save_path):
                raise ValueError("{} already exists.".format(json_save_path))
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            config_info = {
                "Time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "Config": self.config
            }
            log_info_list.append(config_info)
            print(yaml.dump(self.config, default_flow_style=False))

        elif mode == 'train' or mode == 'val':
            print("EPOCH {}".format(log_info_list[-1]['epoch']))
            print("Dice Score: {:.4f}".format(log_info_list[-1]['dice_score']))
        else:
            raise ValueError(f"Invalid mode '{mode}' for save_info function.")

        with open(json_save_path, 'w') as f:
            json.dump(log_info_list, f, indent=2)

    def save_best_model(self, model, epoch):
        dataset_type = self.dataset_config['type']
        model_type = self.model_config['type']
        progress = self.runtime_config['progress']
        model_save_path = 'weights/{}/{}/{}/epoch_{}_best_metrics.pth'.format(dataset_type,
                                                                              model_type,
                                                                              progress,
                                                                              epoch + 1)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # 确保目录存在
        torch.save(model.state_dict(), model_save_path)

    def construct_transform(self, mode):
        assert mode in ['train', 'val'], "mode 只有 train 或 val "

        dataset_config = self.dataset_config
        data_dir = self.dataset_config['root_dir']
        dataset_type = self.dataset_config["type"]
        image_name = self.dataset_config["image_name"]
        mask_name = self.dataset_config["mask_name"]

        mean_key = f"{mode}_mean"
        std_key = f"{mode}_std"
        if mean_key in dataset_config and std_key in dataset_config:
            mean = dataset_config[mean_key]
            std = dataset_config[std_key]
        else:
            print(f"计算{dataset_type}数据集 {mode}_mean 和 {mode}_std 中...")
            structure = f"""
            {dataset_type}
            ├── {mode}
            │   ├── {image_name}
            │   └── {mask_name}
            """
            print(structure)
            mean, std = compute_mean_std(data_dir, image_name, mask_name, mode=mode)
        print("--------------------------------------------")
        print(f"{mode} mean: {mean} \n {mode} std: {std}")
        return Preprocess(mode=mode, mean=mean, std=std)

    def construct_dataset(self, mode):
        image_name = self.dataset_config['image_name']
        mask_name = self.dataset_config['mask_name']
        dataset_dir = self.dataset_config['root_dir']
        dataset_type = self.dataset_config['type']

        transforms = self.construct_transform(mode)
        dataset = LiverDataset(dataset_dir, image_name, mask_name, transform=transforms, mode=mode)

        if mode == 'train':
            print('构建训练集，{}一共有{}张CT图像'.format(dataset_type, len(dataset)))
        elif mode == 'val':
            print('构建验证集，{}一共有{}张CT图像'.format(dataset_type, len(dataset)))
        print("--------------------------------------------")
        return dataset

    def construct_model(self):
        model_type = self.config['model']['type']
        model_params = self.config['model']['params']

        # 动态地从model模块中导入类
        model_module = importlib.import_module(f"models")
        model_class = getattr(model_module, model_type)

        return model_class(**model_params)

    def load_model_weights(self, model, weights_path=None):
        if weights_path is None:
            # 如果未提供weights_path，从config获取
            weights_path = self.model_config.get('weights', None)
            if weights_path is None:
                raise ValueError("未找到权重")
        # 加载权重到模型中
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        return model

    def construct_loss(self):
        loss_type = self.config['loss']['type']
        loss_params = self.config['loss']['params']
        # 动态地从losses模块中导入类
        loss_module = importlib.import_module(f"losses")
        loss_class = getattr(loss_module, loss_type)
        return loss_class(**loss_params)

    def construct_optimizer(self, model_parameters):
        optimizer_type = self.config['optimizer']['type']
        optimizer_params = self.config['optimizer']['params']
        # 使用getattr动态获取优化器类
        optimizer_class = getattr(optim, optimizer_type)
        # 返回创建的优化器实例
        return optimizer_class(model_parameters, **optimizer_params)

    def construct_lr_scheduler(self, optimizer):
        scheduler_config = self.config.get('scheduler')
        if scheduler_config is None:  # 如果scheduler未定义，返回None
            return None
        scheduler_type = scheduler_config['type']
        scheduler_params = scheduler_config['params']
        scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
        return scheduler_class(optimizer, **scheduler_params)

    def construct_early_stop(self):
        # 获取早停配置
        early_stop_config = self.config.get('early_stop')
        # 如果没有配置，则返回None
        if early_stop_config is None:
            return None
        # 获取早停的类型
        early_stop_type = early_stop_config['type']
        # 核实指定的类型是否匹配'EarlyStopping'类
        if early_stop_type != "EarlyStopping":
            raise ValueError(f"不支持的早停类型: {early_stop_type}")
        # 获取早停的参数
        early_stop_params = early_stop_config['params']
        # 直接使用提供的参数实例化EarlyStopping类
        return EarlyStopping(**early_stop_params)





