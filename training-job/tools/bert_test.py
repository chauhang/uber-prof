from argparse import ArgumentParser
from transformers import BertTokenizer, BertModel

from calc_inference_time import calculate_inference_time

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate HF bert inference time")
    parser.add_argument(
        "--warm_up",
        type=int,
        default=10,
        metavar="N",
        help="number of iterations to warm up (default: 10)",
    )

    parser.add_argument(
        "--n_repeat",
        type=int,
        default=100,
        metavar="N",
        help="number of inference request to calculate inference time (default: 100)",
    )

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    text = "bloomberg has reported on the economy"
    encoded_input = tokenizer(text, return_tensors="pt")
    calculate_inference_time(
        model=model,
        inputs={"input_ids": encoded_input.input_ids},
        warm_up=args.warm_up,
        n_repeat=args.n_repeat,
    )