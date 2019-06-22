import argparse
from pathlib import Path
import importlib.util
from evaluation import evaluate_semseg


def import_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('config', type=str, help='Path to configuration .py file')
parser.add_argument('--profile', dest='profile', action='store_true', help='Profile one forward pass')

def eval(config='configs/pyramid.py'):
    conf_path = Path(config)
    conf = import_module(config)
    class_info = conf.dataset_val.class_info

    model = conf.model.cuda()

    for loader, name in conf.eval_loaders:
        iou_acc = evaluate_semseg(model, loader, class_info, observers=conf.eval_observers)
        print(f'{name}: {iou_acc:.2f}')
        # Note: we are expecting only one loader (val)
        return iou_acc


if __name__ == '__main__':
    args = parser.parse_args()
    eval(args.config)
