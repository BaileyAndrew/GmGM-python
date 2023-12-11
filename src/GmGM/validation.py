from typing import *
import numpy as np
import matplotlib.pyplot as plt
from .GmGM import *
from .extras.regularizers import *
from .typing import *
from .generate_data import *

class RunningMeasurer:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.n = 0

    def __call__(self, input):
        self.mean = (self.mean * self.n + input) / (self.n + 1)
        self.var = (self.var * self.n + (input - self.mean) ** 2) / (self.n + 1)
        self.n += 1
    
    def std(self):
        return np.sqrt(self.var)
    
AlgorithmName: TypeAlias = str
RegularizationParameter: TypeAlias = float
PrecisionMatrix: TypeAlias = np.ndarray
AxisName: TypeAlias = str
MetricName: TypeAlias = str

Algorithm: TypeAlias = Callable[
    [Dataset, RegularizationParameter],
    PrecisionMatrix
]

def measure_prec_recall(
    generator: DatasetGenerator,
    algorithms: dict[AlgorithmName, Algorithm],
    Λs: dict[Algorithm, list[float]],
    num_attempts: int,
    num_samples: int = 100,
    *,
    verbose: int = 0,
    give_prior: bool = False
) -> dict[
    AlgorithmName,
    dict[
        AxisName,
        dict[
            MetricName,
            list[float]
        ]
    ]
]:
    """
    Using `generator`, test the performance of `algorithm` on the dataset
    as the regularization parameter Λ varies

    For each Λ, we run `num_attempts` attempts and average the results
    """

    random_alg = list(algorithms.keys())[0]
    output = [None] * len(Λs[random_alg])

    num_λs = len(Λs[random_alg])

    # Create measurers
    measurers = []
    for idx in range(num_λs):
        measurers.append({
            algorithm_name: {
                axis_name: {
                    metric_name: RunningMeasurer()
                    for metric_name in ["precision", "recall"]
                }
                for axis_name in generator.axes
            }
            for algorithm_name in algorithms.keys()
        })

    for i in range(num_attempts):
        if verbose >= 1:
            print(f"Attempt {i+1}/{num_attempts}")

        # Generate a new ground truth
        generator.reroll_Ψs()
        Ψs_true = generator.Ψs

        # Use this new ground truth to generate
        # an input dataset
        dataset = generator.generate(num_samples)

        for idx in range(num_λs):
            if verbose >= 2:
                print(f"λ #{idx}")

            # For each algorithm collect metrics for that
            # algorithm on this dataset
            for algorithm_name, algorithm in algorithms.items():
                if verbose >= 3:
                    print(f"Algorithm: {algorithm_name}")

                # Run algorithm
                if not give_prior:
                    Ψs_pred = algorithm(dataset, generator.structure, Λs[algorithm_name][idx])
                else:
                    Ψs_pred = algorithm(dataset, generator.structure, Λs[algorithm_name][idx], Ψs_true)

                # Get metrics
                Ψs_pred = binarize_matrices(Ψs_pred, eps=1e-3, mode="<Tolerance")
                cm = {
                    axis: generate_confusion_matrices(Ψs_pred[axis], Ψs_true[axis])
                    for axis in generator.axes
                    if axis in Ψs_pred
                }

                precisions = {
                    axis: precision(cm[axis])
                    if axis in cm
                    else 0
                    for axis in generator.axes
                }
                recalls = {
                    axis: recall(cm[axis])
                    if axis in cm
                    else 0
                    for axis in generator.axes
                }

                # Keep count in a Running Measurer
                for axis in generator.axes:
                    measurers[idx][algorithm_name][axis]["precision"](precisions[axis])
                    measurers[idx][algorithm_name][axis]["recall"](recalls[axis])

    for idx in range(num_λs):
        # Get results from the running measurer into a nice dictionary format
        output[idx] = {
            algorithm_name: {
                axis_name: {
                    metric_name: measurers[idx][algorithm_name][axis_name][metric_name].mean
                    for metric_name in ["precision", "recall"]
                }
                for axis_name in generator.axes
            }
            for algorithm_name in algorithms.keys()
        }

        # Add the standated deviations into precision_std and recall_std keys
        for algorithm_name in algorithms.keys():
            for axis_name in generator.axes:
                for metric_name in ["precision", "recall"]:
                    output[idx][algorithm_name][axis_name][metric_name + "_std"] = \
                        measurers[idx][algorithm_name][axis_name][metric_name].std()

    return output

def plot_prec_recall(
    results: dict,
    axis: str,
    generator: DatasetGenerator
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots results of `measure_prec_recall` for `axis`
    """

    # Only keep results for the correct axis
    results = [
        {
            algorithm_name: {
                metric_name: metric_results
                for metric_name, metric_results in algorithm_results[axis].items()
            }
            for algorithm_name, algorithm_results in result.items()
        } for result in results
    ]

    # Now plot the results
    fig, ax = plt.subplots()

    for algorithm_name, _ in results[0].items():
        # Plot each algorithms PR curve
        precisions = [result[algorithm_name]["precision"] for result in results]
        recalls = [result[algorithm_name]["recall"] for result in results]

        ax.plot(recalls, precisions, label=algorithm_name)

        # Add error bounds
        precisions_std = [result[algorithm_name]["precision_std"] for result in results]
        recalls_std = [result[algorithm_name]["recall_std"] for result in results]

        ax.fill_between(
            recalls,
            np.array(precisions) - np.array(precisions_std),
            np.array(precisions) + np.array(precisions_std),
            alpha=0.2
        )

        ax.fill_between(
            np.array(recalls) - np.array(recalls_std),
            precisions,
            precisions,
            alpha=0.2
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    if generator.name is not None:
        ax.set_title(f"Precision-Recall for {generator.name} on {axis}")

    return fig, ax

def generate_confusion_matrices(
    pred: np.ndarray,
    truth: np.ndarray,
    eps: float = 0,
    mode: Literal[
        '<Tolerance',
        'Nonzero',
        'Negative',
        'Mixed',
    ] = '<Tolerance'
) -> np.ndarray:
    """
    `mode`:
        '<Tolerance': x->1 if |x| < eps, else x->0
        'Nonzero': Same as '<Tolerance'
        'Negative': x->1 if x < 0 else x -> 0
        'Mixed': truth is '<Tolerance', pred is 'Negative'
    """
    mode_pred = mode if mode != 'Mixed' else 'Negative'
    mode_truth = mode if mode != 'Mixed' else 'Nonzero'
    pred = binarize_matrix(pred, eps=eps, mode=mode_pred)
    truth = binarize_matrix(truth, eps=0, mode=mode_truth)
    
    # Identity matrices to remove diagonals
    In = np.eye(pred.shape[0])
    
    np.fill_diagonal(pred, 1)
    np.fill_diagonal(truth, 1)
    
    TP: "True positives"
    TP = (pred * truth - In).sum()
    
    FP: "False positives"
    FP = (pred * (1 - truth)).sum()
    
    TN: "True negatives"
    TN = ((1 - pred) * (1 - truth)).sum()
    
    FN: "False negatives"
    FN = ((1 - pred) *  truth).sum()
    
    return np.array([
        [TP, FP],
        [FN, TN]
    ])

def precision(
    cm: np.ndarray
):
    """
    [[TP, FP], [FN, TN]]
    """
    denom = (cm[0, 0] + cm[0, 1])
    if denom == 0:
        return 1
    return cm[0, 0] / denom

def recall(
    cm: np.ndarray
):
    """
    [[TP, FP], [FN, TN]]
    """
    denom = (cm[0, 0] + cm[1, 0])
    if denom == 0:
        return 1
    return cm[0, 0] / denom

def binarize_matrix(
    M: np.ndarray,
    eps: float = 0,
    mode: Literal["Negative", "<Tolerance"] = '<Tolerance'
):
    """
    Returns M but with only ones and zeros
    """
    out = np.empty(M.shape)
    if mode == '<Tolerance' or mode == 'Nonzero':
        out[np.abs(M) <= eps] = 0
        out[np.abs(M) > eps] = 1
    elif mode == 'Negative':
        out[M < 0] = 1
        out[M >= 0] = 0
        
        # In negative mode we need to add diagonals back
        out += np.eye(out.shape[0])
    else:
        raise Exception(f'Invalid mode {mode}')
    return out

def binarize_matrices(
    Ms: dict[Axis],
    eps: float = 0,
    mode: Literal["Negative", "<Tolerance"] = '<Tolerance'
):
    return {
        axis: binarize_matrix(M, eps=eps, mode=mode)
        for axis, M in Ms.items()
    }