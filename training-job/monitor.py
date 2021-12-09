import json
import logging
import os
from argparse import ArgumentParser
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
)


def read_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError("config - {} not found".format(config_path))

    with open(config_path, "r") as fp:
        config_data = json.load(fp)

    return config_data


def find_bad_host(details: dict, expected_error_code: str) -> Optional[dict]:
    for host_details in details:
        if expected_error_code in host_details["value"]:
            logging.debug("Found bad host: {}".format(host_details))
            return host_details
        else:
            return None


def query_for_errors(host: str, config: dict) -> list:

    bad_host_details = []
    for error_code in config.keys():
        url = "http://{}/api/v1/query?query={}".format(host, error_code)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise Exception("Request Failed - {}".format(url))

        try:
            result: dict = json.loads(resp.text)
        except ValueError:
            raise ValueError("Unable to parse response text - {}".format(resp.text))

        if not result["data"]["result"]:
            logging.info("No response found for error code {}".format(error_code))
        else:
            logging.info("Response found for error code {}".format(error_code))
            expected_error_code = str(config[error_code])
            bad_host = find_bad_host(
                details=result["data"]["result"], expected_error_code=expected_error_code
            )
            bad_host_details.append(bad_host)

    return bad_host_details


if __name__ == "__main__":
    parser = ArgumentParser(description="Monitoring argument parser")

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Error config json file path (default: config.json)",
    )

    parser.add_argument(
        "--prometheus_host",
        type=str,
        default="localhost:9090",
        help="prometheus host ip and port (default: localhost:9090)",
    )

    args = vars(parser.parse_args())

    config_data = read_config(config_path=args["config"])

    bad_host_details = query_for_errors(host=args["prometheus_host"], config=config_data)

    if not bad_host_details:
        logging.info("No Bad hosts found")
    else:
        logging.info("Bad host details {}".format(bad_host_details))
