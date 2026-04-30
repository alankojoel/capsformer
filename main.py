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
    TEST = True # set to True if want to perform testing

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
        #if torch.backends.mps.is_available: # remove if not run with Apple Silicon processor
        #    device = "mps"
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        trainer = Trainer(device, **{**array_cfg, **data_cfg, **train_cfg, **rng_cfg, **dir_cfg})
        trainer.run()
    
    if TEST:
        tester = Tester(**{**array_cfg, **data_cfg, **rng_cfg, **dir_cfg})
        #tester.compare_metrics(mode="m", DOAs=np.array([-55.2, 14.4]), DOAs_SOI=None, DOAs_SOI_perturb=False, SNRs=np.array([[-10, -6], [-8, -4], [-6, -2], [-4, 0], [-2, 2], [0, 4], [2, 6], [4, 8], [6, 10], [8, 12], [10, 14]]), MC_trials=500)
