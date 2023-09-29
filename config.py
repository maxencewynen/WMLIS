import os
import argparse
import socket

def setup_config(server=None):
    hostname = socket.gethostname()
    if server in ("jason", "Hatysa", "waldo", "canicula", "cassiopee", "elen") or \
            hostname in ("jason", "Hatysa", "waldo", "canicula", "cassiopee"):
        os.environ["DATA_ROOT_DIR"] = r"/home/mwynen/data/cusl_wml"
        os.environ["MODELS_ROOT_DIR"] = r"/home/mwynen/models/WMLIS"
    elif server == "manneback" or hostname.endswith("cism.ucl.ac.be"):
        os.environ["DATA_ROOT_DIR"] = r"/CECI/home/ucl/elen/mwynen/data/cusl_wml"
        os.environ["MODELS_ROOT_DIR"] = r"/CECI/home/ucl/elen/mwynen/models/WMLIS"
    elif server == "lucia" or hostname.endswith("lucia.cenaero.be"):
        os.environ["DATA_ROOT_DIR"] = r"/gpfs/home/acad/ucl-elen/mwynen/data/cusl_wml"
        os.environ["MODELS_ROOT_DIR"] = r"/gpfs/home/acad/ucl-elen/mwynen/models/WMLIS"
    else:
        raise Exception(f"Server {server} not in known servers. Modify the current file to add it.")

    print(f"$DATA_ROOT_DIR set to {os.environ['DATA_ROOT_DIR']}")
    print(f"$MODELS_ROOT_DIR set to {os.environ['MODELS_ROOT_DIR']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get all command line arguments.')
    # save options
    parser.add_argument('--server', type=str, default=None, help='server name')
    args = parser.parse_args()

    setup_config(server=args.server)

