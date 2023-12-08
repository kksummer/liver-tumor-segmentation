# Liver and Tumor Segmentation Project

本项目是用于肝脏和肿瘤分割的深度学习框架。它提供了训练、评估和预测的功能，使得可以对医学图像进行精确的分割。

## 开始使用

在开始之前，请确保你的系统中已安装了所有必要的依赖项。依赖项可以在项目的`requirements.txt`文件中找到。

### 安装依赖

```bash
pip install -r requirements.txt
```

## 配置文件

项目中使用YAML配置文件来管理训练、评估和预测的参数。你可以在configs目录下找到这些文件，并根据需要进行调整。
* liver_training.yaml - 用于肝脏分割训练的配置文件。
* tumor_training.yaml - 用于肿瘤分割训练的配置文件。
* evaluation.yaml - 用于评估阶段的配置文件。
* prediction.yaml - 用于预测阶段的配置文件。


### 训练模型
要训练肝脏分割模型，请使用以下命令：
```bash
python main.py --liver_train
```

要训练肿瘤分割模型，请使用以下命令：
```bash
python main.py --tumor_train
```

### 评估模型
要评估模型，请使用以下命令：
```bash
python main.py --eval
```
### 预测
要运行预测，请使用以下命令：

```bash
python main.py --pred
```

## 意见反馈
如果你有任何问题或建议，请[发送邮件给我](mailto:ianke@qq.com)
