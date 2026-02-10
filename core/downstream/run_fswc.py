import os
import warnings

from core.downstream.task_helper import setup_downstream_tasks
from core.downstream.downstream import eval_single_task
from core.utils.process_args import process_args
from core.utils.file_utils import save_results

warnings.filterwarnings("ignore", category=DeprecationWarning)


def eval_fswc_loop(args, eval_type_list):

    print("* Evaluating on {}...".format(args["study"]))
    tasks = setup_downstream_tasks(args)
    print("* All datasets to evaluate on = {}".format(list(tasks.keys())))

    MODELS = {
        '{}_{}'.format(args["model"], args["study"]): args["results_dir"],
    }

    for eval_type in eval_type_list:
        print("\033[92m Evaluation Type: {} \033[0m".format(eval_type))
        for exp_name, p in MODELS.items():
            for n, t in tasks.items():
                print('\n* Dataset:', n)
                eval_single_task(args, n, t, p, verbose=False, eval_type=eval_type)

    # save results to results.txt
    print("* Saving results...")
    save_results(args, os.path.join(args["results_dir"], "results.txt"))
