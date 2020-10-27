import os
import numpy as np
import tensorflow as tf
from utils.utils import read_json, make_dot_dic_from_dic, make_dic_from_dot_dic
from utils.utils import str2bool, save_json, set_random_seed
from en_tokenizers import SentencePieceTokenizer
from reformer.reformer import Reformer
from losses import lm_cross_entropy_loss
from dataset import TextSamplerDataset


if __name__ == "__main__":
    config_root = 'config'
    sentencepiece_model_root = 'sentencepiece_models'
    gs_bucket = "gs://saved_reformer_models"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--config_name", type=str, default="reformer.json")
    parser.add_argument("--using_wandb", type=str2bool, default=False)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    args = parser.parse_args()

    data_root = args.data_root
    config_name = args.config_name
    using_wandb = args.using_wandb
    use_rev = args.use_rev

    data_name = data_root.split("/")[-1]

    # settings !
    hp_dict = read_json(f"{config_root}/{config_name}")
    hp = make_dot_dic_from_dic(hp_dict)

    set_random_seed(hp.training.seed)

    sentencepiece_model_name = hp.data.tokenizer
    sentencepiece_model_path = f"{sentencepiece_model_root}/{sentencepiece_model_name}"
    tokenizer = SentencePieceTokenizer(sentencepiece_model_path)

    hp.model.d_k = hp.model.d_model // hp.model.n_heads
    hp.data.vocab_size = tokenizer.vocab_size
    hp_dict = make_dic_from_dot_dic(hp)

    project_name = f"reformer-{data_name}-v2"
    run_name = (f"d_model_{hp.model.d_model}-d_ff-{hp.model.d_ff}-n_layers-{hp.model.n_layers}-"
                f"max_len_{hp.model.max_position_embeddings}-bucket_len_{hp.model.bucket_len}"
                f"-n_rounds_{hp.model.n_rounds}-bsz-{hp.training.batch_size}-lr_{hp.training.lr}")
    run_name = run_name.replace(".", "_")
    print(hp)

    if using_wandb:
        import wandb
        wandb.init(project=project_name, name=run_name, reinit=True,
                   config=hp_dict)

    train_dataset = TextSamplerDataset(hp, data_root, "train", tokenizer)
    print("load model...")
    x = tf.placeholder(dtype=tf.int32, shape=(None, hp.model.max_position_embeddings))
    model = Reformer(hp, use_rev=use_rev)
    logits = model(x)
    loss = lm_cross_entropy_loss(x, logits)

    g_loss_logits = tf.gradients(loss, logits)
    g_loss_logits = g_loss_logits[0]
    grads_and_vars = model.compute_gradients(logits, g_loss_logits)

    opt = tf.keras.optimizers.Adam(hp.training.lr)
    apply = opt.apply_gradients(grads_and_vars)
    print("model is loaded!")

    os.makedirs(f"saved_model/reformer-{data_name}/{run_name}", exist_ok=True)
    save_json(f'saved_model/reformer-{data_name}/{run_name}/config.json', hp_dict)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        epochs = hp.training.epochs
        steps_per_epoch = hp.training.steps_per_epoch

        template = "\r{:.2f} epochs, loss : {:.4f}, bit per dim : {:.4f}"
        for epoch in range(epochs):
            total_loss = 0

            for step, data in enumerate(train_dataset):
                _, l = sess.run([apply, loss], feed_dict={x: data})

                total_loss += l

                if (step + 1) % 1 == 0:  # hard coding
                    n_epoch = epoch + (step + 1) / steps_per_epoch
                    mean_loss = total_loss / (step + 1)
                    bit_per_dim = mean_loss / np.log(2)
                    print(template.format(
                        n_epoch, mean_loss, bit_per_dim), end="")
                if (step + 1) == steps_per_epoch:
                    break

            print(template.format(
                n_epoch, mean_loss, bit_per_dim))
            if using_wandb:
                wandb.log({"loss": mean_loss, "bit_per_dim": bit_per_dim, "epochs": epoch,
                           "steps": (epoch+1)*steps_per_epoch})

            if epoch % 5 == 0:
                model.save_weights(filepath=f'saved_model/reformer-{data_name}/{run_name}/{run_name}')

