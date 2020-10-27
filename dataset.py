import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextSamplerDataset(Sequence):
    def __init__(self, hp, root, phase, tokenizer):
        super(TextSamplerDataset, self).__init__()
        self.phase = phase
        self.max_len = hp.model.max_position_embeddings
        self.tokenizer = tokenizer
        self.batch_size = hp.training.batch_size
        with open(f"{root}/{phase}.txt") as f:
            self.text = f.read()
        self.on_epoch_end()

    def __len__(self):
        return len(self.text)  # sufficient large value

    def __getitem__(self, idx):
        rand_starts = np.random.randint(0, len(self.text)-self.max_len-1, self.batch_size)
        x = [self.tokenizer.encode(self.text[i:i+self.max_len+1].strip()) for i in rand_starts]
        x = pad_sequences(x, maxlen=self.max_len,
                          padding='post', truncating='post')
        return x


if __name__ == "__main__":
    import time
    from utils.utils import read_json, make_dot_dic_from_dic
    from en_tokenizers import SentencePieceTokenizer
    hp = make_dot_dic_from_dic(read_json("config/reformer.json"))
    data_root = 'data/enwik8'
    phase = 'dev'
    tokenizer = SentencePieceTokenizer("sentencepiece_models/enwik8-bpe.model")

    dataset = TextSamplerDataset(hp, data_root, phase, tokenizer)
    dataset.max_len = 32
    for i in dataset:
        for j in i:
            # print(j)
            # print(tokenizer.decode(j.tolist()), end='\r')
            print(j[:10], end='\r')
            time.sleep(1)

