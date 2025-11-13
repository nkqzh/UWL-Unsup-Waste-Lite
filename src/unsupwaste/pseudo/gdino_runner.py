# src/unsupwaste/pseudo/gdino_runner.py

from dataclasses import dataclass
from typing import List, Dict
import torch, yaml
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import load_image


@dataclass
class Det:
    xyxy: list
    score: float
    text: str


class GDRunner:
    def __init__(self, prompts: List[str], device='cpu',
                 box_thr=0.3, text_thr=0.25):
        self.prompts = prompts
        self.device = device
        self.box_thr = box_thr
        self.text_thr = text_thr

        # 直接使用 OWL-ViT 作为开放词汇检测器
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        ).to(device)
        self.backend = "owlvit"

    @classmethod
    def from_cfg(cls, cfg_path, device='cpu'):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cls(
            cfg.get('text_prompts', ['trash']),
            device=device,
            box_thr=cfg.get('box_threshold', 0.3),
            text_thr=cfg.get('text_threshold', 0.25),
        )

    @torch.no_grad()
    def detect(self, image_path: str) -> List[Dict]:
        # 单路径：OWL-ViT 推理
        image = load_image(image_path)
        inputs = self.processor(
            text=[self.prompts],
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model(**inputs)
        target_sizes = torch.tensor(
            [image.size[::-1]],
            device=self.model.device
        )
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.box_thr,
            target_sizes=target_sizes
        )[0]

        out = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            out.append({
                "xyxy": box.tolist(),
                "score": float(score),
                "text": self.prompts[int(label)],
            })
        return out
