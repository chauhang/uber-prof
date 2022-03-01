import subprocess
import os
import time
import json
from argparse import ArgumentParser


def run_command(cmd: str):
    print("Running command: {}".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    raw_output, raw_err = p.communicate()
    return raw_output


def wait_for_slurm_log(job_id: str):
    print("*" * 50 + "\nLog generation\n" + "*" * 50)
    slurm_log_file: str = "slurm-{}.out".format(job_id)
    for _ in range(30):
        print("Waiting for slurm log file - {} to be created".format(slurm_log_file))
        if os.path.exists(slurm_log_file):
            print("Slurm log file - {}found\n".format(slurm_log_file))
            return slurm_log_file
        else:
            time.sleep(10)
    else:
        raise Exception("Slurm log file {} not found".format(slurm_log_file))


def wait_for_training_start(log_file):
    print("*" * 50 + "\nTraining\n" + "*" * 50)
    for _ in range(30):
        print("Waiting for training to get started")
        with open(log_file) as fp:
            data = fp.read()
        if "Epoch 0" in data:
            print("Training start found\n")
            return True
        else:
            time.sleep(10)
    else:
        return False


def start_training(slurm_script):
    print("*" * 50 + "\nInvoking slurm job\n" + "*" * 50)
    cmd = "sbatch {}".format(slurm_script)
    output = run_command(cmd)
    if not isinstance(output, bytes) or "Submitted batch job" not in output.decode("utf-8"):
        raise Exception("Unable to start training with command: {}".format(cmd))
    else:
        job_id = output.decode("utf-8").split(" ")[-1].rstrip("\n")
        print("Retreived Job ID: ", job_id)
        print("Slurm job started successfully \n")
        slurm_log_file: str = wait_for_slurm_log(job_id=job_id)
        training_status = wait_for_training_start(log_file=slurm_log_file)
        if not training_status:
            raise Exception("Training start not detected")
    return slurm_log_file, job_id


def inject_error(host: str, gpu_id: int, f_param: int, v_param: int, name: str):
    print("*" * 50 + "\nInjecting Error - {} \n".format(name) + "*" * 50)
    cmd = "dcgmi test --host {} --inject --gpuid {} -f {} -v {}".format(
        host, gpu_id, f_param, v_param
    )
    response = run_command(cmd)
    print(response)
    print("\n")


def check_dcgm_error(log_file: str, error_msg: str):
    print("*" * 50 + "\nListing for violation\n" + "*" * 50)
    for _ in range(10):
        print("Waiting for violation msg - {}".format(error_msg))
        with open(log_file) as fp:
            data = fp.read()
            if error_msg in data:
                print("Violation found\n")
                return True
            else:
                time.sleep(10)
    else:
        raise Exception("Violations not found")


def read_dcgm_errors_file(file: str):
    if not os.path.exists(file):
        raise FileNotFoundError("DCGM error file - {} not found".format(file))
    else:
        with open(file) as fp:
            data = json.load(fp)

    return data


def cancel_dcgm_job(job_id):
    print("*" * 50 + "\nCancelling Job - {}\n".format(job_id) + "*" * 50)
    cmd = "scancel {}".format(job_id)
    output = run_command(cmd)
    print(output)

    cmd = "squeue"
    for _ in range(30):
        output = run_command(cmd)
        print("Squeue output: ", output)
        if job_id in output.decode("utf-8").split("\n")[1]:
            print("Job still running - {}".format(output))
            time.sleep(10)
        else:
            print("Job Cancelled successfully\n")
            break
    else:
        raise Exception("Unable to cancel the existing job")


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

    parser.add_argument(
        "--dcgm_error_file",
        type=str,
        default="dcgm_errors.json",
        help="Path to dcgm errors file",
    )

    args = vars(parser.parse_args())
    dcgm_errors_content = read_dcgm_errors_file(file=args["dcgm_error_file"])

    for dcgm_error in dcgm_errors_content:
        name = dcgm_error["name"]
        if "XID" in name or "ECC" in name or "NVLink" in name:
            slurm_log_file, job_id = start_training(slurm_script=args["slurm_script"])
            inject_error(
                host=args["compute_host"],
                gpu_id=args["gpu_id"],
                f_param=dcgm_error["f_param"],
                v_param=dcgm_error["v_param"],
                name=dcgm_error["name"],
            )
            check_dcgm_error(log_file=slurm_log_file, error_msg=dcgm_error["msg"])
            cancel_dcgm_job(job_id=job_id)
