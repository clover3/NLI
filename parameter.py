import argparse
import os
parser = argparse.ArgumentParser()
pa = parser.add_argument
pa("--datapath", type=str, default="data")
pa("--batch_size", type=int, default="128")
pa("--lstm_size", type=int, default="1024")
pa("--max_sequence", type=int, default="100")
pa("--frange", type=int, default="0")
pa("--char_vocab_size", type=int, default=400)
pa("--out_channel_dims", type=str, default="100")
pa("--filter_heights", type=str, default="5")
pa("--char_emb_size", type=int, default=8, help="char emb size")
pa("--char_in_word_size", type=int, default=16, help="number of chars in word")
pa("--char_out_size", type=int, default=100, help="char out size")
pa("--use_char_emb", action='store_true', help="use character level info")
pa("--syntactical_features", action='store_true', help="if to use synt. features")
pa("--keep_rate", type=float, default=1.0,
   help="Keep rate for dropout in the model")
pa("--use_gpu", type=int, default=2,
   help="The index of the gpu to use for tensorflow")

args = parser.parse_args()

path_dict = {
    "training_mnli": os.path.join(
        args.datapath, "multinli_0.9", "multinli_0.9_train.jsonl"),
    "shared_mnli": os.path.join(args.datapath, "shared.jsonl"),
    "dev_matched": os.path.join(
        args.datapath, "multinli_0.9", "multinli_0.9_dev_matched.jsonl"),
    "dev_mismatched": os.path.join(
        args.datapath, "multinli_0.9", "multinli_0.9_dev_mismatched.jsonl"),
    "test_matched": os.path.join(
        args.datapath, "multinli_0.9",
        "multinli_0.9_test_matched_unlabeled.jsonl"),
    "test_mismatched": os.path.join(
        args.datapath, "multinli_0.9",
        "multinli_0.9_test_mismatched_unlabeled.jsonl"),
    "training_snli": os.path.join(
        args.datapath, "snli_1.0", "snli_1.0_train.jsonl"),
    "dev_snli": os.path.join(
        args.datapath, "snli_1.0", "snli_1.0_dev.jsonl"),
    "test_snli": os.path.join(
        args.datapath, "snli_1.0", "snli_1.0_test.jsonl"),
    "embedding_data_path": os.path.join(args.datapath, "glove.840B.300d.txt"),
}
