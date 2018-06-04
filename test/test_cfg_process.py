import os, sys
sys.path.append("..")
from utils import cfg_process
import unittest
import tempfile

try:
    import ConfigParser
    config = ConfigParser.ConfigParser()
except ImportError:
    import configparser
    config = configparser.ConfigParser()


CONFIG_TEMPLATE_DIR = '../task_config_template.cfg'


class ConfigTest(unittest.TestCase):
    def test_change_num_gpu_and_epochs(self):
        args = {
            "task_name": "resnet50_cifar10_symbolic",
            "num_gpus": 3,
            "epochs": 3,
            "metrics_postfix": "daily",
        }
        tmp = os.path.join(tempfile.gettempdir(), "task_config.cfg")
        cfg_process.generate_cfg(CONFIG_TEMPLATE_DIR, tmp, **args)
        config.read(tmp)
        self.assertEqual(int(config.get(args["task_name"], "num_gpus")), args["num_gpus"])
        cmd = config.get(args["task_name"], "command_to_execute")
        self.assertEqual(cmd.strip(), "python image_classification/image_classification.py "
                                      "--model resnet50_v1 --dataset cifar10"
                                      " --mode symbolic --gpus {0} --epochs {1} "
                                      "--log-interval 50".format(args["num_gpus"], args["epochs"]))
        os.remove(tmp)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ConfigTest)
    unittest.TextTestRunner(verbosity=2).run(suite)