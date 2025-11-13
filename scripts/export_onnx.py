# export_onnx.py
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--opset', type=int, default=12)
    ap.add_argument('--dynamic', action='store_true')
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.export(format='onnx', opset=args.opset, dynamic=args.dynamic)

if __name__ == '__main__':
    main()