import pydcgm
import pytest
import time
import os
import logger
import dcgm_structs
import dcgm_structs_internal
import queue
import dcgm_fields
import logging as log
from ctypes import *
from utils import (
    wait_for_slurm_log,
    wait_for_training_start,
    start_training,
    inject_error,
    check_dcgm_error,
    read_dcgm_errors_file,
    cancel_dcgm_job,
    get_checkpoint_file,
    remove_checkpoint_files,
)

DCGM_ERROR_FILE = "dcgm_errors.json"
HOST = "queue0-st-queue0-p38xlarge-1"
GPU_ID = 0
SLURM_SCRIPT = "bert.slurm"


def create_c_callback(queue=None):
    @CFUNCTYPE(None, c_void_p)
    def c_callback(data):
        if queue:
            # copy data into a python struct so that it is the right format and is not lost when "data" var is lost
            callbackData = dcgm_structs.c_dcgmPolicyCallbackResponse_v1()
            memmove(addressof(callbackData), data, callbackData.FieldsSizeof())
            queue.put(callbackData)

    return c_callback


@pytest.fixture
def setup():
    remove_checkpoint_files()
    dcgm_errors_content = read_dcgm_errors_file(file=DCGM_ERROR_FILE)

    slurm_log_file, job_id = start_training(SLURM_SCRIPT)
    assert job_id is not None
    assert wait_for_training_start(log_file=slurm_log_file, search_string="Epoch 2")
    yield slurm_log_file, dcgm_errors_content, job_id
    cancel_dcgm_job(job_id)


def test_dcgm(setup):
    _, _, job_id = setup
    log.debug("Intializing DCGM")
    dcgmHandle = pydcgm.DcgmHandle(ipAddress="queue0-st-queue0-p38xlarge-1")
    dcgmSystem = dcgmHandle.GetSystem()
    dcgmSystem.GetAllGroupIds()
    supportedGPUs = dcgmSystem.discovery.GetAllSupportedGpuIds()
    group = dcgmSystem.GetGroupWithGpuIds("allgpus", supportedGPUs)
    callbackQueue = queue.Queue()
    c_callback = create_c_callback(callbackQueue)
    log.debug("Registering callback")
    group.policy.Register(dcgm_structs.DCGM_POLICY_COND_XID, c_callback, None)

    log.debug("Injecting error")
    inject_error(
        host=HOST,
        gpu_id=GPU_ID,
        name="XID",
        f_param=230,
        v_param=49,
    )
    log.debug("{} Wait for callback {}".format("*" * 50, "*" * 50))
    try:
        callbackData = callbackQueue.get(timeout=30)
        log.debug("Callback triggered. Callback data: {}".format(callbackData))
        cancel_dcgm_job(job_id=job_id)
    except queue.Empty:
        assert False, "Callback never happened"

    log.debug("{} Resuming training {}".format("*" * 50, "*" * 50))
    slurm_log_file, job_id = start_training(SLURM_SCRIPT)
    assert job_id is not None
    checkpoint = get_checkpoint_file("epoch=1-step=*.ckpt")
    assert os.path.exists(checkpoint)
    assert wait_for_training_start(
        log_file=slurm_log_file,
        search_string="Restoring states from the checkpoint path at {}".format(checkpoint),
    )
    assert wait_for_training_start(log_file=slurm_log_file, search_string="Epoch 2")

    log.debug("{} Test successful. Cancelling existing job {}".format("*" * 50, "*" * 50))
    cancel_dcgm_job(job_id)
