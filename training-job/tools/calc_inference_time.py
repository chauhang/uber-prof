import timeit

import torch
from tqdm import tqdm


def calculate_inference_time(
    model, inputs, warm_up: int, n_repeat: int, profiler: bool = True, profiler_args: dict = None
):
    for _ in tqdm(range(warm_up)):
        if isinstance(inputs, dict):
            _ = model(**inputs)
        else:
            _ = model(inputs)

    print("Calculating average time taken for inference")

    mean_inference_time = 0
    if profiler:
        if profiler_args is None:
            profiler_args = {
                "activities": [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                "on_trace_ready": torch.profiler.tensorboard_trace_handler("./logs"),
            }
        with torch.profiler.profile(**profiler_args) as prof:
            for _ in tqdm(range(n_repeat)):
                if isinstance(inputs, dict):
                    mean_inference_time += timeit.timeit(lambda: model(**inputs), number=1)
                else:
                    mean_inference_time += timeit.timeit(lambda: model(inputs), number=1)
                prof.step()
    else:
        for _ in tqdm(range(n_repeat)):
            if isinstance(inputs, dict):
                mean_inference_time += timeit.timeit(lambda: model(**inputs), number=1)
            else:
                mean_inference_time += timeit.timeit(lambda: model(inputs), number=1)

    mean_inference_time = round(mean_inference_time / n_repeat, 2)
    print("Avg time taken for inference: ", mean_inference_time)

    return mean_inference_time
