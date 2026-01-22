import argparse

from pathlib import Path 

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModel

from video_refiner.util import instantiate_from_config

def load_video_refiner(config_path: Path, weights_path: Path):
    video_refiner_config = OmegaConf.load(config_path)
    video_refiner = instantiate_from_config(video_refiner_config["model"])
    ckpt = torch.load(weights_path, weights_only=False, map_location="cpu")["module"]
    for key in list(ckpt.keys()):
        if "_forward_module" in key:
            ckpt[key.replace("_forward_module.", "")] = ckpt[key]
        del ckpt[key]
    missing, unexpected = video_refiner.load_state_dict(ckpt, strict=False)
    print(f"Missing keys: {missing}")
    return video_refiner

def prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_refiner_config",
        type=Path,
        default="../configs/video_refiner/inference.yaml"
    )
    parser.add_argument(
        "--video_refiner_weights",
        type=Path,
        default="../checkpoints/video_refiner/refiner_model.pt"
    )
 
if __name__ == "__main__":
    args = prepare_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    video_refiner = load_video_refiner(args.video_refiner_config, args.video_refiner_weights).to(device).eval()
    video_refiner.token_decoder = AutoModel.from_pretrained("turing-motors/Terra", subfolder=tokenizer_name, trust_remote_code=True).to(device).eval()

