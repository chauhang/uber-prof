import subprocess
import os
import time
from argparse import ArgumentParser


def run_command(cmd: str):
    print("Running command {}".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    raw_output, raw_err = p.communicate()
    return raw_output


def wait_for_slurm_log(job_id: str):
    slurm_log_file: str = "slurm-{}.out".format(job_id)
    for _ in range(30):
        print("Waiting for slurm log file - {} to be created".format(slurm_log_file))
        if os.path.exists(slurm_log_file):
            print("Slurm log file found")
            return slurm_log_file
        else:
            time.sleep(10)
    else:
        raise Exception("Slurm log file {} not found".format(slurm_log_file))


def wait_for_training_start(log_file):
    for _ in range(30):
        print("Waiting for epoch 0 to be started")
        with open(log_file) as fp:
            data = fp.read()
        if "Epoch 0" in data:
            print("Training start found")
            return True
        else:
            time.sleep(10)
    else:
        return False


def start_training(slurm_script):
    cmd = "sbatch {}".format(slurm_script)
    output = run_command(cmd)
    if not isinstance(output, bytes) or "Submitted batch job" not in output.decode("utf-8"):
        raise Exception("Unable to start training with command: {}".format(cmd))
    else:
        job_id = output.decode("utf-8").split(" ")[-1].rstrip("\n")
        print("Job ID: ", job_id)
        slurm_log_file: str = wait_for_slurm_log(job_id=job_id)
        training_status = wait_for_training_start(log_file=slurm_log_file)
        if not training_status:
            raise Exception("Training start not detected")

    return slurm_log_file


def inject_error(host: str, gpu_id: int, f_param: int, v_param: int):
    cmd = "dcgmi test --host {} --inject --gpuid {} -f {} -v {}".format(
        host, gpu_id, f_param, v_param
    )
    response = run_command(cmd)
    print(response)


def check_dcgm_error(log_file: str):
    for _ in range(10):
        print("Waiting for violation")
        with open(log_file) as fp:
            data = fp.read()
            if "violated policy manager values" in data:
                print("Violation found")
                return True
            else:
                time.sleep(10)
    else:
        raise Exception("Violations not found")


if __name__ == "__main__":
    parser = ArgumentParser(description="Bert-News Classifier Example")

    parser.add_argument(
        "--slurm_script",
        type=str,
        default="bert.slurm",
        help="Path to slurm script",
    )

    parser.add_argument(
        "--compute_host",
        type=str,
        required=True,
        help="IP or fully qualified domain of compute host",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id",
    )

    args = vars(parser.parse_args())
    slurm_log_file = start_training(slurm_script=args["slurm_script"])
    inject_error(host=args["compute_host"], gpu_id=args["gpu_id"], f_param=319, v_param=48)
    check_dcgm_error(log_file=slurm_log_file)
