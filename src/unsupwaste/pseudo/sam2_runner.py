class SAM2Runner:
    def __init__(self, device='cpu'):
        self.device = device
        try:
            from sam2.build_sam import build_sam2 # noqa: F401
            self.ok = True
        except Exception:
            self.ok = False

    def refine(self, image_path, dets):
        # 预留：可在此调用 SAM2 根据框生成掩码 → 再写 YOLO-seg
        return dets