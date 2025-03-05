# EHOP: A Dataset of Everyday NP-Hard Optimization Problem

This repository contains the code and dataset used in the experiments described [here](https://coli-saar.github.io/ehop).

## Package Structure

Each problem's package has the following structure:

- `model`: Representations of problem instances and solutions for the problem.
- `generator`: Functionality for generating problem instances.
- `llm`: LLM-based solvers for the problem.
- `symbolic`: Symbolic solvers for the problem.
- `alt`: Alternate (typically suboptimal) solvers for the problem.

Each problem creates subclasses of classes defined in the `base` directory. Consult the docstrings and comments there to better understand how components interact.

## Replicating Experiments

### Requirements

Ensure you are using Python version >= 3.10, and install the necessary packages by running `pip install -r requirements.txt`.

To run evaluations using a Llama model, one must have access to the given model through Hugging Face, be logged in, and have the model downloaded (the page for Llama-3.1-70B-Instruct is [here](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) and provides documentation for getting access and downloading the model).
To run evaluations using a GPT model, one must have one's OpenAI API key [set as the appropriate environment variable](https://platform.openai.com/docs/quickstart#create-and-export-an-api-key).

### Running Experiments

Given a problem (`graph_coloring`, `knapsack`, or `traveling_salesman`), a model (`gpt` or `llama`), and a (sub-)dataset (`random` or `hard`), the corresponding experiment can be executed by running the following command in the command line:

```{bash}
python main.py configs/<problem>/<model>-<dataset>.jsonnet
```

For example, to run GPT-4o on the knapsack component of the hard dataset, one would run the following:

```{bash}
python main.py configs/knapsack/gpt-hard.jsonnet
```

When running such experiments, please note the following:

- To run Llama experiments, one must provide the path to the downloaded model in the appropriate section of each Llama config file.
- It is important to note that instances in the random and hard datasets follow the same naming scheme and that results for a given (model $\times$ dataset) combination will be sent to a consistent filepath. Thus, one should not simultaneously run experiments for the random and hard components of the EHOP dataset using the same model. Instead, run them sequentially, and organize/rename the result file from the first experiment so that new results are not appended to it. Any simultaneous experiments that use distinct (model $\times$ dataset) combinations should produce results in distinct files and not produce any issues.

## Customization

- **Experiment Configuration Files:** To run custom experiments, simply create your own `.jsonnet` file using the same format used in the provided examples.
- **Dataset Creation:** To generate custom versions of the dataset, use the `generator.py` files in a given problem's directory to generate new problem instances.
