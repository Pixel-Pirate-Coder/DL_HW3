import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
import random


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data_file: str, train: bool = True, sp_model_prefix: str = None,
                 vocab_size: int = 2000, normalization_rule_name: str = "nmt_nfkc_cf",
                 model_type: str = "bpe", max_length: int = 256, pad_id: int = 3):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(sp_model_prefix + ".model"):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id=pad_id
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + ".model")

        with open(data_file, encoding="utf-8") as file:
            texts = file.readlines()

        # Включаем разделение на обучающую и валидационную выборки в методе __init__
        self.train_texts, self.val_texts = self.split_train_val(texts, self.VAL_RATIO, self.TRAIN_VAL_RANDOM_SEED)
        self.texts = self.train_texts if train else self.val_texts
        self.indices = [self.text2ids(text) for text in self.texts] # FIX: Поменял texts на text
        
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
                self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    @staticmethod
    def split_train_val(data, val_ratio, random_seed):
        random.seed(random_seed)
        random.shuffle(data)
        split_idx = int(len(data) * val_ratio)
        return data[:-split_idx], data[-split_idx:]

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, "Expected tensor of shape (length, ) or (batch_size, length)"
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        indices = self.indices[item]  # Получаем индексы текста

        # Добавляем специальные токены и обрезаем до максимальной длины
        indices = [self.bos_id] + indices + [self.eos_id]
        indices = indices[:self.max_length]

        # Дополняем до максимальной длины
        padded_indices = indices + [self.pad_id] * (self.max_length - len(indices))

        return torch.tensor(padded_indices), len(indices)  # Возвращаем тензор с заполненными индексами и их фактическую длину
