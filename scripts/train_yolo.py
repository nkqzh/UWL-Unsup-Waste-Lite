import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True) # configs/dataset_*.yaml
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--project', default='runs/yolo11')
    p.add_argument('--train-dir', default=None) # 若指定伪标目录
    args = p.parse_args()

    model = YOLO(args.model)
    overrides = dict(project=args.project, imgsz=args.imgsz, epochs=args.epochs)
    if args.train_dir:
        overrides.update(train=args.train_dir)
    model.train(data=args.data, **overrides)

if __name__ == '__main__':
    main()