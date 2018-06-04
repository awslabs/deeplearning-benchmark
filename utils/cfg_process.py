import os
import re

try:
    import ConfigParser
    config = ConfigParser.ConfigParser()
    new_config = ConfigParser.RawConfigParser()
except ImportError:
    import configparser
    config = configparser.ConfigParser()
    new_config = configparser.RawConfigParser()


def generate_cfg(cfg_template, cfg_path, **infra_spec):
    """ Read infrastructure specific parameters and genreate a new task config

    :param: cfg_template string
        file path to the cfg template
    :param: cfg_path string
        file path of generated cfg file
    :param: infra_spec dict
        dictionary containing infrastructure specific parameters
    """
    if os.path.isfile(cfg_path):
        os.remove(cfg_path)
    config.read(cfg_template)
    selected_task = infra_spec['task_name']
    new_config.add_section(selected_task)
    for name, value in config.items(selected_task):
        if name != "num_gpus" and name != "command_to_execute":
            new_config.set(selected_task, name, config.get(selected_task, name))
        elif "num_gpus" in infra_spec and name == "num_gpus":
            new_config.set(selected_task, name, infra_spec[name])
 # check for overrides, if any            
        elif name == "command_to_execute":
            cmd = config.get(selected_task, name)
            if "num_gpus" in infra_spec and infra_spec["num_gpus"] > 0:
                cmd = re.sub("--gpus \d", "--gpus %d" % infra_spec["num_gpus"], cmd)
            elif "num_gpus" in infra_spec:
                cmd = re.sub("--gpus \d", "", cmd)
            if "epochs" in infra_spec and infra_spec["epochs"] > 0:
                cmd = re.sub("--epochs \d+", "--epochs %d" % infra_spec["epochs"], cmd)
            if "kvstore" in infra_spec:
                cmd = re.sub("--kvstore device", "--kvstore %s" % infra_spec["kvstore"], cmd)
            if "dtype" in infra_spec:
                cmd = re.sub("--dtype float32", "--dtype %s" % infra_spec["dtype"], cmd)
    
                
            new_config.set(selected_task, name, cmd)
    with open(cfg_path, 'w') as cfg:
        new_config.write(cfg)
