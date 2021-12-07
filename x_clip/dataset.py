import torch
from pathlib import Path
from PIL import UnidentifiedImageError, Image
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    def __init__(self,
                 preprocess,
                 folder,
                 bpe_tokenizer,
                 truncate_text=True,
                 context_length=256):
        super().__init__()
        path = Path(folder)
        self.truncate_text = truncate_text
        self.context_length = context_length

        text_files = [*path.glob("**/*.txt")]
        text_files = {text_file.stem: text_file for text_file in text_files}
        if len(text_files) == 0:
            raise ValueError("No text files found in folder.")
        image_files = [
            *path.glob("**/*.png"),
            *path.glob("**/*.jpg"),
            *path.glob("**/*.jpeg"),
            *path.glob("**/*.bmp"),
        ]
        image_files = {
            image_file.stem: image_file
            for image_file in image_files
        }
        if len(image_files) == 0:
            raise ValueError("No image files found in folder.")

        keys = None
        join = lambda new_set: new_set & keys if keys is not None else new_set
        keys = join(text_files.keys())
        keys = join(image_files.keys())

        self.keys = list(keys)
        self.tokenizer = bpe_tokenizer
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.image_transform = preprocess

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        try:
            image_file = self.image_files[key]
            image_tensor = self.image_transform(
                Image.open(image_file).convert("RGB"))
        except (UnidentifiedImageError, OSError):
            print(f"Failed to load image {image_file}. Skipping.")
            return None  # return None to be filtered in the batch collate_fn

        text_file = self.text_files[key]
        caption = text_file.read_text()
        tokenized_text = self.tokenizer.tokenize(
            caption,
            context_length=self.context_length,
            truncate_text=self.truncate_text).squeeze(0)
        return tokenized_text, image_tensor