import logging
import os
from functools import wraps, partial
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))  # Go up 5 levels

print("Current directory:", current_dir)
print("Project root directory:", project_root)

# Add project root to the Python path
if project_root not in sys.path:
    sys.path.append(project_root)
from deps.experiment_launcher.experiment_launcher.utils import save_args, start_wandb, create_results_dir


def wrapper_single_experiment(exp_func, save_args_yaml=False, use_logging=False,
                              make_dirs_with_seed=True, print_exp_args=False):
    @wraps(exp_func)
    def wrapper(
        # Function arguments
        *args,
        **kwargs
    ):
        # Make results directory
        create_results_dir(kwargs, make_dirs_with_seed)

        # Setup logging
        if use_logging:
            logging.basicConfig(level=logging.INFO,
                                filename=os.path.join(kwargs['results_dir'], "logfile"),
                                # stream=sys.stdout,
                                filemode="a+",
                                format="%(asctime)-15s %(levelname)-8s %(message)s")

        # Save arguments
        save_args(kwargs['results_dir'], kwargs,
                  git_repo_path='./', save_args_as_yaml=save_args_yaml, print_exp_args=print_exp_args)

        # Start WandB
        wandb_run = start_wandb(**kwargs)

        # Run the experiment
        exp_func(*args, **kwargs)

        # Clean up
        wandb_run.finish()

    return wrapper


single_experiment = partial(wrapper_single_experiment)
single_experiment_flat = partial(wrapper_single_experiment, make_dirs_with_seed=False)
single_experiment_yaml = partial(wrapper_single_experiment, save_args_yaml=True, use_logging=True, print_exp_args=False)
single_experiment_flat_yaml = partial(wrapper_single_experiment, make_dirs_with_seed=False, save_args_yaml=True)
