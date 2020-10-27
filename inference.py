import time
import numpy as np
from utils.utils import read_json, make_dot_dic_from_dic
from en_tokenizers import SentencePieceTokenizer
from reformer.reformer import Reformer

if __name__ == "__main__":
    sentencepiece_model_root = 'sentencepiece_models'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--n_chars", type=int, default=200)
    args = parser.parse_args()

    model_dir = args.model_dir
    n_chars = args.n_chars

    text = input("text를 입력하세요 : ")
    original_text = text

    model_name = model_dir.split('/')[-1]
    config_path = f'{model_dir}/config.json'
    model_weight_path = f"{model_dir}/{model_name}"

    hp_dict = read_json(config_path)
    hp = make_dot_dic_from_dic(hp_dict)

    sentencepiece_model_name = hp.data.tokenizer
    sentencepiece_model_path = f"{sentencepiece_model_root}/{sentencepiece_model_name}"
    tokenizer = SentencePieceTokenizer(sentencepiece_model_path)

    hp.model.d_k = hp.model.d_model // hp.model.n_heads
    hp.data.vocab_size = tokenizer.vocab_size
    max_len = hp.model.max_position_embeddings

    model = Reformer(hp)
    model.load_weights(model_weight_path)

    print('\n\n\n\n')
    for _ in range(n_chars):
        # print("\r" + text)  # colab에서는 안보일때가 있
        print(text)
        ids = tokenizer.encode(text)

        if text[-1] == " ":
            ids += [4]
        if ids[0] == 4:
            ids = ids[1:]

        front_text, ids = ids[:-max_len + 1], ids[-max_len + 1:]

        next_token_idx = len(ids) - 1

        padded_ids = ids[:max_len] + [0] * (max_len - len(ids))

        inp = np.array([padded_ids])
        logits = model.predict(inp)[0]

        pred_token_ids = np.argmax(logits, axis=1).tolist()

        next_char_id = pred_token_ids[next_token_idx]
        next_char = tokenizer.decode([next_char_id])
        if next_char:
            text += next_char
        else:
            text += " "

        time.sleep(0.3)

    print("original text :", original_text)
    print("generated text :", text)
