import time
from typing import cast, Iterable, Iterator, List, TypeVar
import itertools
import datasets
import numpy as np
import pandas as pd
import torch
from chronos.chronos_bolt_icl import ChronosBoltICLPipeline
import sys
sys.path.append('/Users/a0s0lwe/Desktop/my-codebase/chronos-forecasting/scripts/training')
import data_provider.icl_data_factory as IclDataFactory

import fev

T = TypeVar("T")
def batcher(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    it: Iterator[T] = iter(iterable)
    def get_batch():
        return list(itertools.islice(it, batch_size))
    return iter(get_batch, [])


def predict_window(
    window: fev.EvaluationWindow,
    model: ChronosBoltICLPipeline,
    context_length: int,
    quantile_levels: list[float],
    batch_size: int,
    return_mean: bool,
    min_examples: int,
    max_examples: int,
    min_past_for_examples: int,
    use_exogenous: bool = True, # whether to use both past_dynamic_columns and/or known_dynamic_columns
    use_past_dynamic: bool = True, # whether to use past_dynamic_columns (only if use_exogenous is True)
    use_known_dynamic: bool = True, # whether to use known_dynamic_columns (only if use_exogenous is True)
) -> tuple[datasets.DatasetDict, float]:
    if 0.5 not in quantile_levels:
        quantile_levels.append(0.5)
    past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=True)

    quantiles_all = []
    mean_all = []
    window_inference_time = 0.0
    for batch_inputs in batcher(zip(past_data, future_data), batch_size=batch_size):
        batch_past = datasets.Dataset.from_list([item[0] for item in batch_inputs])
        batch_future = datasets.Dataset.from_list([item[1] for item in batch_inputs])
        icl_batch_inputs, _, _, _ = IclDataFactory.convert_past_future_datasets_to_icl_input(
            past_data=batch_past, future_data=batch_future,
            target_columns=["target"], past_dynamic_columns=window.past_dynamic_columns, 
            known_dynamic_columns=window.known_dynamic_columns, 
            prediction_length=window.horizon, context_length=context_length,
            min_examples=min_examples, max_examples=max_examples, 
            min_past_for_examples=min_past_for_examples,
            use_exogenous=use_exogenous, use_past_dynamic=use_past_dynamic,
            use_known_dynamic=use_known_dynamic,
        )
        batch_start_time = time.monotonic()
        output_quantiles, output_mean = model.predict_quantiles(
            icl_inputs=[_input for _input in icl_batch_inputs],
            prediction_length=window.horizon,
            quantile_levels=quantile_levels,
        )
        window_inference_time += time.monotonic() - batch_start_time
        quantiles_all.append(output_quantiles.cpu().numpy())
        mean_all.append(output_mean.cpu().numpy())

    quantiles_np = np.concatenate(quantiles_all, axis=0)  # [num_items, horizon, num_quantiles]
    mean_np = np.concatenate(mean_all, axis=0)  # [num_items, horizon]
    if return_mean: # use mean as the point forecast
        point_forecast = mean_np  # [num_items, horizon]
    else: # use median as the point forecast
        point_forecast = quantiles_np[:, :, quantile_levels.index(0.5)]  # [num_items, horizon]
    predictions_dict = {"predictions": point_forecast}
    for idx, level in enumerate(quantile_levels):
        predictions_dict[str(level)] = quantiles_np[:, :, idx]

    predictions_for_window = fev.utils.combine_univariate_predictions_to_multivariate(
        datasets.Dataset.from_dict(predictions_dict), target_columns=window.target_columns
    )

    return predictions_for_window, window_inference_time


def predict_with_model(
    task: fev.Task,
    model_name: str = "icl",
    batch_size: int = 32,
    device_map: str = "cpu",
    torch_dtype: str = torch.float32,
    min_examples: int = 1,
    max_examples: int = 3,
    min_past_for_examples: int = 4,
    use_exogenous: bool = True, 
    use_past_dynamic: bool = True, 
    use_known_dynamic: bool = True, 
) -> tuple[list[datasets.DatasetDict], float, dict]:
    pipeline = ChronosBoltICLPipeline.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch_dtype,
    )

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows():
        predictions, window_inference_time = predict_window(
            window,
            model=pipeline,
            context_length=pipeline.model.chronos_config.context_length,
            quantile_levels=task.quantile_levels,
            batch_size=batch_size,
            return_mean=task.eval_metric in ["MSE", "RMSE", "RMSSE"],
            min_examples=min_examples, max_examples=max_examples, min_past_for_examples=min_past_for_examples,
            use_exogenous=use_exogenous, use_past_dynamic=use_past_dynamic, use_known_dynamic=use_known_dynamic,
        )
        predictions_per_window.append(predictions)
        inference_time += window_inference_time

    extra_info = {
        "model_config": {
            "model_name": model_name,
            "batch_size": batch_size,
            "device_map": device_map,
            "torch_dtype": str(torch_dtype),
        }
    }

    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "ICL"
    model_path = "/Users/a0s0lwe/Desktop/my-codebase/chronos-forecasting/output/chronos_bolt_icl_init_model"
    num_tasks = None  # replace with `num_tasks = None` to run on all tasks
    min_examples = 1
    max_examples = 3
    min_past_for_examples = 4
    use_exogenous = True
    use_past_dynamic = True
    use_known_dynamic = True
    use_seasonality = False

    model_name = (
        model_name + "_" + str(min_examples) + "." + str(max_examples) + "." + 
        str(min_past_for_examples) + "_" + str(use_exogenous)[0] + "." + 
        str(use_past_dynamic)[0] + "." + str(use_known_dynamic)[0] + "." +
        str(use_seasonality)[0] 
    )
    benchmark = fev.Benchmark.from_yaml(
        "/Users/a0s0lwe/Desktop/fev/benchmarks/example/tasks.yaml"
    )
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(
            task, model_name=model_path,
            min_examples=min_examples, max_examples=max_examples, min_past_for_examples=min_past_for_examples,
            use_exogenous=use_exogenous, use_past_dynamic=use_past_dynamic, use_known_dynamic=use_known_dynamic,
        )
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name=model_name,
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv(model_name + ".csv", index=False)
