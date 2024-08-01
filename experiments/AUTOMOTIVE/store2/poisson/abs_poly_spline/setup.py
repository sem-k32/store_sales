import os
import yaml


if __name__ == "__main__":
    # write root dir into params.yaml
    with open("params.yaml", "r") as f:
        params_dict = yaml.load(f, yaml.CLoader)
    params_dict["WD"] = os.environ["WORKSPACE_DIR"]
    with open("params.yaml", "w") as f:
        yaml.dump(f, params_dict, yaml.CDumper)
