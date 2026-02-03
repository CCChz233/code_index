from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from method.indexing.core.block import Block


class BlockDataset(Dataset):
    def __init__(self, blocks: List[Block], tokenizer, max_length: int, ir_context_tokens: int = 256):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ir_context_tokens = ir_context_tokens

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b = self.blocks[idx]
        is_ir_function = b.block_type == "ir_function" and bool(b.function_text or b.context_text)
        if is_ir_function:
            input_ids, attention_mask, orig_len, truncated = self._encode_ir_function(b)
        else:
            input_ids, attention_mask, orig_len, truncated = self._encode_default(b)
        return input_ids, attention_mask, orig_len, truncated, is_ir_function

    def _pad_to_max_length(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            pad_ids = torch.full((pad_len,), pad_id, dtype=input_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=0)
        elif input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        return input_ids, attention_mask

    def _encode_default(self, b: Block) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        text = f"file path: {b.file_path}\nlines: {b.start}-{b.end}\n\n{b.content}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        orig_len = int(input_ids.size(0))
        truncated = orig_len >= self.max_length
        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = self._pad_to_max_length(input_ids, attention_mask)
        return input_ids, attention_mask, orig_len, truncated

    def _encode_ir_function(self, b: Block) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        context_text = b.context_text or ""
        function_text = b.function_text or b.content

        if hasattr(self.tokenizer, "num_special_tokens_to_add") and callable(
            getattr(self.tokenizer, "num_special_tokens_to_add")
        ):
            special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        else:
            special_tokens = 2
        max_function_len = max(self.max_length - special_tokens, 0)

        context_ids = []
        if context_text:
            context_ids = self.tokenizer(
                context_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.ir_context_tokens,
            )["input_ids"]

        function_ids = self.tokenizer(
            function_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_function_len,
        )["input_ids"]

        orig_len = len(context_ids) + len(function_ids) + special_tokens

        remaining_context = max(self.max_length - special_tokens - len(function_ids), 0)
        context_ids = context_ids[:remaining_context]

        combined_ids = context_ids + function_ids
        if hasattr(self.tokenizer, "build_inputs_with_special_tokens") and callable(
            getattr(self.tokenizer, "build_inputs_with_special_tokens")
        ):
            input_ids = self.tokenizer.build_inputs_with_special_tokens(combined_ids)
        else:
            # TokenizersBackend 等无此方法时，手动加 [CLS] + content + [SEP]
            cls_id = getattr(self.tokenizer, "cls_token_id", None) or getattr(
                self.tokenizer, "bos_token_id", None
            )
            sep_id = getattr(self.tokenizer, "sep_token_id", None) or getattr(
                self.tokenizer, "eos_token_id", None
            )
            if cls_id is None:
                cls_id = sep_id if sep_id is not None else 0
            if sep_id is None:
                sep_id = 0
            input_ids = [cls_id] + combined_ids + [sep_id]
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        input_ids, attention_mask = self._pad_to_max_length(input_ids, attention_mask)

        truncated = orig_len >= self.max_length
        return input_ids, attention_mask, orig_len, truncated
