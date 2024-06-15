import torch
from torch import nn
from typing import Type
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
from dataset import TextDataset # FIX: просто dataset заменил на src.dataset (в соответствии со своей структурой)


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(
            num_embeddings=dataset.vocab_size,
            embedding_dim=embed_size,
            padding_idx=dataset.pad_id
        )
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, dataset.vocab_size) # FIX: убрал len() у dataset.vocab_size

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        embeds = self.embedding(indices)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(packed_embeds)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        logits = self.linear(outputs)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 0.1) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        
        tokens = self.dataset.text2ids(prefix)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        embeds = self.embedding(tokens)

        if embeds.numel() != 0:
            output, hidden = self.rnn(embeds)

            new_tokens = Categorical(logits=self.linear(output[:, -1]) / temp).sample()
            tokens = torch.cat([tokens, new_tokens.unsqueeze(0)], dim=1)

            while tokens.shape[1] < self.max_length:
                if new_tokens.item() == self.dataset.eos_id:
                    break
                embeds = self.embedding(new_tokens.unsqueeze(0))
                output, hidden = self.rnn(embeds, hidden)
                new_tokens = Categorical(logits=self.linear(output[:, -1])).sample()
                tokens = torch.cat([tokens, new_tokens.unsqueeze(0)], dim=1)

            return self.dataset.ids2text(tokens.squeeze())
        else:
            return ''

