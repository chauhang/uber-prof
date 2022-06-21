import json
import time
from utils import run_command


class DcgmMonitoring:
    def __init__(self):
        self.job_queue = None
        self.job_index = None
        self.compute_nodes: list = []
        self.iteration: int = 10
        self.interval: int = 10
        self.is_health_watch_enabled = False

    def get_compute_hosts(self):
        try:
            nodes: str = self.job_queue["jobs"][self.job_index]["nodes"]
            # queue0 - st - queue0 - p3xlarge - [1 - 2]
            base_name: str = nodes.split("[")[0]
            node_range: str = nodes.split("[")[1]
            node_range: str = node_range.split("]")[0]
            start_range, end_range = node_range.split("-")
            for i in range(int(start_range), int(end_range) + 1):
                self.compute_nodes.append("{}{}".format(base_name, i))
        except KeyError as e:
            print("No Jobs found.. Returning")

    def is_job_running(self):
        for i in range(len(self.job_queue["jobs"])):
            if self.job_queue["jobs"][i]["job_state"] == "RUNNING":
                print("Job found at index: ", i)
                self.job_index = i
                return True
        return False

    def run_health_checks(self, host: str):
        health_check = "dcgmi health --host {} -g 1 -c -j".format(host)
        # for _ in range(self.iteration):
        result: bytes = run_command(health_check)
        result_dict: dict = json.loads(result.decode("utf-8"))
        print("Monitoring health status...")
        print(result_dict)
        if result_dict["body"]["Overall Health"]["value"] == "Healthy":
            print("Healthy node.. Waiting for {} seconds".format(self.interval))
        else:
            print("!!!!!!!!!! Warning: Unhealthy status: {}".format(json.dumps(result_dict)))
        time.sleep(self.interval)

    def enable_health_watch(self, host: str):
        cmd = "dcgmi health --host {} -g 1 -s mpi ".format(host)
        try:
            resp = run_command(cmd)
            print(resp)
            if resp.decode("utf-8").rstrip() == "Health monitor systems set successfully.":
                return True
            else:
                return False
        except Exception as e:
            print("Unable to enable health watch: ", str(e))
            return False

    def run_continous_monitoring(self):
        while True:
            result: str = run_command("squeue --json").decode("utf-8")
            self.job_queue: dict = json.loads(result)

            if not self.is_job_running():
                print("No jobs found in the queue. Stopping..")
                break
            else:
                print(
                    "Script - {} found. Continue monitoring".format(
                        self.job_queue["jobs"][self.job_index]["command"]
                    )
                )
                self.get_compute_hosts()
                if len(self.compute_nodes) == 0:
                    print("Unable to find compute node details")
                    break
                else:
                    if not self.is_health_watch_enabled:
                        for host in self.compute_nodes:
                            self.enable_health_watch(host=host)
                        self.is_health_watch_enabled = True
                    print("Compute nodes: ", self.compute_nodes)
                    for host in self.compute_nodes:
                        print("running health check for host: ", host)
                        self.run_health_checks(host=host)
                # run_health_checks()


if __name__ == "__main__":
    DcgmMonitoring().run_continous_monitoring()
