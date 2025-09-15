import yaml
import warnings

from src.models import *
from src.training import *
from src.data_generator import *
from src.evaluation import *

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__  == "__main__":

    GENERATE_DATA = False # set to True if want to generate training and validation data
    TRAIN = False # set to True if want to train the model
    TEST = False # set to True if want to perform testing

    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    array_cfg = config["array"]
    data_cfg = config["data"]
    train_cfg = config["train"]
    rng_cfg = config["rng"]
    dir_cfg = config["dirs"]
    
    if GENERATE_DATA:
        generator = DataGenerator(**{**array_cfg, **data_cfg, **rng_cfg, **dir_cfg})
        generator.generate_data()
    
    if TRAIN:
        if torch.backends.mps.is_available: # remove if not run with Apple Silicon processor
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        trainer = Trainer(device, **{**array_cfg, **data_cfg, **train_cfg, **rng_cfg, **dir_cfg})
        trainer.run()
    
    if TEST:
        tester = Tester(**{**array_cfg, **data_cfg, **rng_cfg, **dir_cfg})
        #tester.compare_metrics(mode="m", DOAs=np.array([-4.2, 14.4]), DOAs_SOI=None, DOAs_SOI_perturb=False, SNRs=np.array([[-10, -7], [-8, -5], [-6, -3], [-4, -1], [-2, 1], [0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-61.1, -10.3, 6.9, 40]), DOAs_SOI=np.array([6.9]), DOAs_SOI_perturb=False, SNRs=np.array([[-6, -8, -10, -8], [-4, -6, -8, -6], [-2, -4, -6, -4], [0, -2, -4, -2], [2, 0, -2, 0], [4, 2, 0, 2], [6, 4, 2, 4], [8, 6, 4, 6], [10, 8, 6, 8], [12, 10, 8, 10], [14, 12, 10, 12]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-61.1, -10.3, 6.9, 40]), DOAs_SOI=np.array([6.9]), DOAs_SOI_perturb=True, SNRs=np.array([[-6, -8, -10, -8], [-4, -6, -8, -6], [-2, -4, -6, -4], [0, -2, -4, -2], [2, 0, -2, 0], [4, 2, 0, 2], [6, 4, 2, 4], [8, 6, 4, 6], [10, 8, 6, 8], [12, 10, 8, 10], [14, 12, 10, 12]]), MC_trials=500)

