# infer_onnx.py
import argparse, time
import onnxruntime as ort
import numpy as np, cv2

def letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]; r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw, dh = dw//2, dh//2
    img = cv2.resize(img, new_unpad)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--source', required=True)
    args = ap.parse_args()

    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'], sess_options=so)

    im = cv2.imread(args.source)[:, :, ::-1]
    im0 = im.copy()
    im = letterbox(im, (640,640))
    im = im.astype(np.float32)/255.0
    im = np.transpose(im, (2,0,1))[None]

    t0 = time.time()
    out = sess.run(None, {sess.get_inputs()[0].name: im})[0]
    print(f"Latency: {(time.time()-t0)*1000:.1f} ms")
    # TODO: 根据导出头解析 boxes/scores/classes 并可视化