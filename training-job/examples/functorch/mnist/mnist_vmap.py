import torch
import time
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser
from functorch import combine_state_for_ensemble, vmap
from mnist import Net


def load_model(args):
    model = Net()
    print("Loading MNIST model from {}".format(args.model_path))
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

    start_time = time.time()
    predictions2 = [model(img_tensor) for model in models]
    print("Prediction: ", predictions2)
    print("Time taken for prediction without vmap: ", time.time() - start_time)
    print("\n")

    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    # print([p.size(0) for p in params])

    start_time = time.time()
    predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, img_tensor)
    print("Prediction: ", predictions2_vmap)
    print("Time taken for prediction with vmap: ", time.time() - start_time)


if __name__ == "__main__":

    parser = ArgumentParser("MNIST Inference")

    parser.add_argument(
        "--image_path",
        type=str,
        default="1.png",
        help="Path to image file",
    )

    parser.add_argument(
        "--num_models",
        type=int,
        default=2,
        help="Number of models to predict",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="mnist_cnn.pt",
        help="Path to model serialized file",
    )

    args = parser.parse_args()

    model = load_model(args=args)
    img_tensor = transform_image(args=args)
    run_sample_prediction(model=model, img_tensor=img_tensor)
    serve_multiple_models(args=args, model=model, img_tensor=img_tensor)
