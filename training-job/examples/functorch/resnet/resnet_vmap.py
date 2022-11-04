import timeit
from argparse import ArgumentParser

import torch
from PIL import Image
from functorch import combine_state_for_ensemble, vmap
from model import ImageClassifier
from torchvision import transforms
from tqdm import tqdm


def load_model(args):
    model = ImageClassifier()
    print("Loading resnet model from {}".format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    return model


def transform_image(args):
    print("Loading image from {}".format(args.image_path))
    image = Image.open(args.image_path)
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
        ]
    )
    img_tensor = transform(image)

    img_tensor = img_tensor.unsqueeze(dim=0)

    print("Converted image to shape {}".format(img_tensor.shape))

    return img_tensor


def run_sample_prediction(model, img_tensor):

    print("Running Sample prediction")

    model.eval()

    print(model)

    result = model(img_tensor.float()).argmax(dim=1)

    print("Prediction: ", result)
    print("\n\n")


def serve_multiple_models(args, model, img_tensor):
    print("Serving multiple models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [model.to(device) for _ in range(args.num_models)]

    img_tensor = img_tensor.float().to(device)

    print("Running Warm up")
    WARM_UP = 100
    for _ in tqdm(range(WARM_UP)):
        _ = [model(img_tensor) for model in models]

    N_REPEAT = 1000
    mean_hf_time = 0
    for _ in tqdm(range(N_REPEAT)):
        mean_hf_time += timeit.timeit(lambda: [model(img_tensor) for model in models], number=1)

    print("Avg time taken for prediction without vmap: ", round(mean_hf_time / N_REPEAT, 2))
    print("\n")

    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    print("Running Warm up")
    WARM_UP = 100
    for _ in tqdm(range(WARM_UP)):
        _ = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, img_tensor)

    N_REPEAT = 1000
    mean_hf_time = 0
    for _ in tqdm(range(N_REPEAT)):
        mean_hf_time += timeit.timeit(
            lambda: vmap(fmodel, in_dims=(0, 0, None))(params, buffers, img_tensor), number=1
        )

    print("Avg time taken for prediction with vmap: ", round(mean_hf_time / N_REPEAT, 2))


if __name__ == "__main__":

    parser = ArgumentParser("Resnet Inference")

    parser.add_argument(
        "--image_path",
        type=str,
        default="kitten.jpg",
        help="Path to image file",
    )

    parser.add_argument(
        "--num_models",
        type=int,
        default=10,
        help="Number of models to predict",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="resnet18-f37072fd.pth",
        help="Path to model serialized file",
    )

    args = parser.parse_args()

    model = load_model(args=args)
    img_tensor = transform_image(args=args)
    run_sample_prediction(model=model, img_tensor=img_tensor)
    serve_multiple_models(args=args, model=model, img_tensor=img_tensor)
