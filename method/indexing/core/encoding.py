from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from method.core.embedding import extract_last_hidden_state, pool_hidden_states
from method.indexing.core.block import Block
from method.indexing.core.dataset import BlockDataset


def embed_blocks_coderank(
    blocks: List[Block],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    ir_context_tokens: int = 256,
    pooling: str = "first_non_pad",
    show_progress: bool = False,
    progress_desc: str = "Blocks",
) -> torch.Tensor:
    """CodeRankEmbed 专用的 embedding 函数，使用 AutoModel 后端。"""
    ds = BlockDataset(blocks, tokenizer, max_length, ir_context_tokens)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outs = []

    pbar = None
    if show_progress:
        pbar = tqdm(
            total=len(ds),
            desc=progress_desc,
            unit="blk",
            leave=False,
            dynamic_ncols=True,
            position=1,
        )

    try:
        with torch.no_grad():
            for input_ids, attn_mask, orig_len, truncated, is_ir in loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                token_embeddings = extract_last_hidden_state(outputs)
                sent_emb = pool_hidden_states(token_embeddings, attn_mask, pooling=pooling)

                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
                outs.append(sent_emb.cpu())

                if pbar is not None:
                    pbar.update(input_ids.size(0))
    finally:
        if pbar is not None:
            pbar.close()

    return torch.cat(outs, dim=0)


def embed_blocks_sfr(
    blocks: List[Block],
    model,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """SFR 专用的 embedding 函数，使用 SentenceTransformer 后端。"""
    ds = BlockDataset(blocks, tokenizer, max_length, ir_context_tokens=256)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outs = []

    with torch.no_grad():
        for input_ids, attn_mask, orig_len, truncated, is_ir in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            outputs = model.forward({
                "input_ids": input_ids,
                "attention_mask": attn_mask,
            })

            if isinstance(outputs, dict):
                sent_emb = outputs.get("sentence_embedding") or outputs.get("embeddings")
            elif isinstance(outputs, torch.Tensor):
                sent_emb = outputs
            else:
                raise RuntimeError(f"Unexpected output type: {type(outputs)}")

            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())

    return torch.cat(outs, dim=0)
