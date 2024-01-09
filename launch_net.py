# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import shutil
import submitit

import pathlib
from shutil import copytree, ignore_patterns, rmtree

from detectron2.engine import default_argument_parser, launch

def parse_args():
    d2_arg_parser = default_argument_parser()
    parser = argparse.ArgumentParser(
        "Submitit launcher for D2-based Guided Distillation", parents=[d2_arg_parser], add_help=False
    )

    parser.add_argument(
        "--partition", default="learnlab", type=str, help="Partition where to submit"
    )
    parser.add_argument("--timeout", default=60 * 24 * 3, type=int, help="Duration of the job")
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--resume-from",
        default="",
        type=str,
        help=(
            "Weights to resume from (.*pth file) or a file (last_checkpoint) that contains "
            + "weight file name from the same directory"
        ),
    )
    parser.add_argument("--resume-job", default="", type=str, help="resume training from the job")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument("--name", default="D2_DETR", type=str, help="Name of the jobs")
    parser.add_argument("--mail", default="", type=str,
                        help="Email this user when the job finishes if specified")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(os.path.join(os.getcwd(), "logs"))
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def copy_codebase_to_experiment_dir(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)
    code_dir = pathlib.Path(experiment_dir) / "code"
    
    # remove code dir to avoid duplicates.
    if os.path.exists(code_dir):
        rmtree(code_dir)
    os.makedirs(code_dir, exist_ok=False)
    
    current_code_dir = pathlib.Path(__file__).parent.resolve()
    print(f"copying {current_code_dir} to {code_dir}")
    copytree(current_code_dir, code_dir, dirs_exist_ok=True, ignore=ignore_patterns('data', 'checkpoints', 'output', 'weights', '*.png', '*.jpg', '*.jpeg', '*.pth', '*.pkl', 'core', 'training-runs', 'weights', 'ckpt', 'logs', 'datasets', '.git'))
    return code_dir


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train_net

        socket_name = os.popen("ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        # not sure if the next line is really affect anything
        # os.environ["NCCL_SOCKET_IFNAME"] = socket_name

        hostname_first_node = os.popen(
            "scontrol show hostnames $SLURM_JOB_NODELIST"
        ).read().split("\n")[0]
        dist_url = "tcp://{}:12399".format(hostname_first_node)
        print("We will use the following dist url: {}".format(dist_url))

        self._setup_gpu_args()
        launch(
            train_net.main,
            self.args.num_gpus,
            num_machines=self.args.num_machines,
            machine_rank=self.args.machine_rank,
            dist_url=dist_url,
            args=(self.args,),
        )
        
    def checkpoint(self):
        import submitit

        self.args.resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        output_dir = str(self.args.output_dir).replace("%j", str(job_env.job_id))
        # self.output_dir = output_dir
        # copy_codebase_to_experiment_dir(output_dir)

        # copy codebase directory.
        
        if self.args.resume_from != "":
            if job_env.global_rank == 0:
                p = os.path.join(output_dir, "output")
                os.makedirs(p, exist_ok=True)

                if self.args.resume_from.endswith(".pth"):
                    weights_file = self.args.resume_from
                else:
                    with open(self.args.resume_from, "r") as f:
                        weights_filename = f.read().strip()
                    weights_file = os.path.join(
                        os.path.dirname(self.args.resume_from), weights_filename
                    )
                print("Copy weights file {} to {}".format(weights_file, p))
                shutil.copy(weights_file, p)
                with open(os.path.join(p, "last_checkpoint"), 'w') as f:
                    f.write(os.path.basename(weights_file))
            self.args.resume = True
            self.args.resume_from = ""
        self.args.opts.extend(["OUTPUT_DIR", os.path.join(output_dir, "output")])
        print(self.args)

        self.args.machine_rank = job_env.global_rank
        print(f"Process rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        job_dir = get_shared_folder()
        args.job_dir = job_dir / "%j"
    else:
        job_dir = args.job_dir

    if args.resume_job != "":
        print("Resuming job {}".format(args.resume_job))
        job_dir_to_resume = os.path.join(str(args.job_dir).replace("%j", args.resume_job), "output")
        args.config_file = os.path.join(job_dir_to_resume, "config.yaml")
        args.resume_from = os.path.join(job_dir_to_resume, "last_checkpoint")
        if not os.path.isfile(args.resume_from):
            args.resume_from = ""
        name = os.popen(
            'sacct -j {} -X --format "JobName%200" -n'.format(args.resume_job)
        ).read().strip()
        args.name = "{}_resumed_from_{}".format(name, args.resume_job)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.num_gpus
    nodes = args.num_machines
    partition = args.partition
    timeout_min = args.timeout
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=1,
        cpus_per_task=10 * num_gpus_per_node,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.name)
    if args.mail:
        executor.update_parameters(
            additional_parameters={'mail-user': args.mail, 'mail-type': 'END'})

    args.output_dir = args.job_dir
    trainer = Trainer(args)
    job = executor.submit(trainer)

    # copy_codebase_to_experiment_dir(args.job_dir)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
