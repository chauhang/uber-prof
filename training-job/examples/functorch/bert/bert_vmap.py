import timeit

import torch
from functorch import combine_state_for_ensemble, vmap
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained("bert-base-cased")
    return model, tokenizer


def run_sample_prediction(model, tokenizer, encoded_input):
    output = model(**encoded_input)
    print(output.last_hidden_state)
    print(output.last_hidden_state.shape)


def serve_multiple_models(model, tokenizer, encoded_input):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_OF_MODELS = 5
    models = [model.to(device) for _ in range(NUM_OF_MODELS)]

    encoded_input = tokenizer(text, return_tensors="pt")
    encoded_input = encoded_input.to(device)

    print("Running Warm up")
    WARM_UP = 100
    for _ in tqdm(range(WARM_UP)):
        _ = [model(**encoded_input, return_dict=False) for model in models]

    N_REPEAT = 1000
    mean_hf_time = 0
    for _ in tqdm(range(N_REPEAT)):
        mean_hf_time += timeit.timeit(
            lambda: [model(**encoded_input, return_dict=False) for model in models], number=1
        )

    print("Avg time taken for prediction without vmap: ", round(mean_hf_time / N_REPEAT, 2))
    print("\n")

    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    print("Running Warm up")
    WARM_UP = 100
    for _ in tqdm(range(WARM_UP)):
        _ = vmap(fmodel)(params, buffers, **encoded_input, return_dict=False)

    N_REPEAT = 1000
    mean_hf_time = 0
    for _ in tqdm(range(N_REPEAT)):
        mean_hf_time += timeit.timeit(
            lambda: vmap(fmodel)(params, buffers, **encoded_input, return_dict=False), number=1
        )

    print("Avg time taken for prediction with vmap: ", round(mean_hf_time / N_REPEAT, 2))


if __name__ == "__main__":
    model, tokenizer = load_model()
    text = "bloomberg has reported on the economy"
    encoded_input = tokenizer(text, return_tensors="pt")

    print("\n Running Sample prediction")

    run_sample_prediction(model=model, tokenizer=tokenizer, encoded_input=encoded_input)
    print("\n")

    serve_multiple_models(model=model, tokenizer=tokenizer, encoded_input=encoded_input)
