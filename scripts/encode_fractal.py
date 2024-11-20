import os
import json
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
import json
import pickle

from models.multimodal_encoder.t5_encoder import T5Embedder

GPU = 0
MODEL_PATH = "/data/models/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
TARGET_DIR = "data/datasets/openx_embod/fractal20220817_data"

def encode_instruction(text_encoder, tokenizer, instruction, device):
    tokens = tokenizer(
        instruction, return_tensors="pt",
        padding="longest",
        truncation=True
    )["input_ids"].to(device)

    tokens = tokens.view(1, -1)
    with torch.no_grad():
        pred = text_encoder(tokens).last_hidden_state.detach().cpu().squeeze(0)
    
    return instruction, pred

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    buffer_path = Path(config['dataset']['buf_path'])
    instructions = []
    save_dict = {}

    # chunks = list(buffer_path.iterdir())
    # total_chunks = len(chunks)
    # for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
    #     if chunk.is_dir():
    #         for file in chunk.iterdir():
    #             if file.suffix == ".json":
    #                 with open(file, "r") as f:
    #                     data = json.load(f)
    #                     import ipdb; ipdb.set_trace()
    #                     if data['instruction'] not in instructions:
    #                         instructions.append(data['instruction'])

    path = "data/datasets/openx_embod/fractal20220817_data/instructions_emb.pkl"
    with open(path, "rb") as f:
        instructions = pickle.load(f)
    
    for instruction, embed in tqdm(instructions.items()):
        if instruction != "":
            if instruction[-1] != '.':
                instruction += '.'
            if instruction[0].islower():
                instruction = instruction[0].upper() + instruction[1:]
        instr, emb = encode_instruction(text_encoder, tokenizer, instruction, device)
        save_dict[instr] = emb

    save_path = Path(TARGET_DIR) / "instructions_emb_upper.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)

if __name__ == "__main__":
    main()
    

