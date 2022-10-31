import time

import torch
from functorch import combine_state_for_ensemble, vmap
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
    models = [model.to(device) for _ in range(2)]

    encoded_input = tokenizer(text, return_tensors="pt")
    encoded_input = encoded_input.to(device)

    start_time = time.time()
    predictions2 = [model(encoded_input.input_ids, return_dict=False) for model in models]
    print("Prediction: ", predictions2[0][0].shape, predictions2[1][0].shape)
    print("Time taken for prediction without vmap: ", time.time() - start_time)
    print("\n")

    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    start_time = time.time()

    predictions2_vmap = vmap(fmodel)(params, buffers, **encoded_input, return_dict=False)
    # print("Prediction: ", predictions2_vmap)
    print("Prediction: ", predictions2_vmap[0][0].shape, predictions2_vmap[1][0].shape)
    print("Time taken for prediction with vmap: ", time.time() - start_time)


if __name__ == "__main__":
    model, tokenizer = load_model()
    text = "bloomberg has reported on the economy"
    encoded_input = tokenizer(text, return_tensors="pt")

    print("\n Running Sample prediction")

    run_sample_prediction(model=model, tokenizer=tokenizer, encoded_input=encoded_input)
    print("\n")

    serve_multiple_models(model=model, tokenizer=tokenizer, encoded_input=encoded_input)
