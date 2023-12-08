import argparse
from src.train import train
from src.evaluate import evaluation
from src.predict import prediction
from configs.get_config import *


def main(args):
    # 加载配置文件
    liver_cfg_path = 'configs/liver_training.yaml'
    tumor_cfg_path = 'configs/tumor_training.yaml'
    eval_cfg_path = 'configs/evaluation.yaml'
    pred_cfg_path = 'configs/prediction.yaml'

    if args.liver_train:
        liver_config = Config(liver_cfg_path)
        train(liver_config)
    elif args.tumor_train:
        tumor_config = Config(tumor_cfg_path)
        train(tumor_config)
    elif args.eval:
        eval_config = Config(eval_cfg_path)
        evaluation(eval_config)
    elif args.pred:
        pred_config = Config(pred_cfg_path)
        prediction(pred_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run different pipeline stages.')
    parser.add_argument('--liver_train', action='store_true', help='Run training stage.')
    parser.add_argument('--tumor_train', action='store_true', help='Run training stage.')
    parser.add_argument('--eval', action='store_true', help='Run evaluation stage.')
    parser.add_argument('--pred', action='store_true', help='Run prediction stage.')
    args = parser.parse_args()
    main(args)
