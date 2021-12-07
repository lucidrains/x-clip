# yttm tokenizer
import torch
from pathlib import Path
import youtokentome as yttm


class YttmTokenizer:
    def __init__(self, bpe_path=None):
        bpe_path = Path(bpe_path)
        assert bpe_path.exists(
        ), f'BPE json path {str(bpe_path)} does not exist'
        tokenizer = yttm.BPE(model=str(bpe_path))
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()

    def decode(self, tokens, pad_tokens=set()):
        if torch.is_tensor(tokens): tokens = tokens.tolist()
        return self.tokenizer.decode(tokens, ignore_ids=pad_tokens.union({0}))

    def encode(self, texts):
        encoded = self.tokenizer.encode(texts, output_type=yttm.OutputType.ID)
        return list(map(torch.tensor, encoded))

    def tokenize(self, texts, context_length=256, truncate_text=False):
        if isinstance(texts, str): texts = [texts]
        all_tokens = self.encode(texts)
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}"
                    )
            result[i, :len(tokens)] = tokens.clone().detach()

        return result