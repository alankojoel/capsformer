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

        #tester.compare_metrics(mode="m", DOAs=np.array([-61.1, -10.3, 6.9, 40]), DOAs_SOI=np.array([6.9]), DOAs_SOI_perturb=False, SNRs=np.array([[-6, -8, -10, -8], [-4, -6, -8, -6], [-2, -4, -6, -4], [0, -2, -4, -2], [2, 0, -2, 0], [4, 2, 0, 2], [6, 4, 2, 4], [8, 6, 4, 6], [10, 8, 6, 8], [12, 10, 8, 10], [14, 12, 10, 12]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-45.02, -30.02, -20.02, -3]), DOAs_SOI=np.array([-45.02]), DOAs_SOI_perturb=False, SNRs=np.array([[-10, -12, -14, -16], [-8, -10, -12, -14], [-6, -8, -10, -12], [-4, -6, -8, -10], [-2, -4, -6, -8], [0, -2, -4, -6], [2, 0, -2, -4], [4, 2, 0, -2], [6, 4, 2, 0], [8, 6, 4, 2], [10, 8, 6, 4]]), MC_trials=500)

        #tester.compare_metrics(mode="m", DOAs=np.array([-61.1, -10.3, 6.9, 40]), DOAs_SOI=np.array([6.9]), DOAs_SOI_perturb=True, SNRs=np.array([[-6, -8, -10, -8], [-4, -6, -8, -6], [-2, -4, -6, -4], [0, -2, -4, -2], [2, 0, -2, 0], [4, 2, 0, 2], [6, 4, 2, 4], [8, 6, 4, 6], [10, 8, 6, 8], [12, 10, 8, 10], [14, 12, 10, 12]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-45.02, -30.02, -20.02, -3]), DOAs_SOI=np.array([-45.02]), DOAs_SOI_perturb=True, SNRs=np.array([[-10, -12, -14, -16], [-8, -10, -12, -14], [-6, -8, -10, -12], [-4, -6, -8, -10], [-2, -4, -6, -8], [0, -2, -4, -6], [2, 0, -2, -4], [4, 2, 0, -2], [6, 4, 2, 0], [8, 6, 4, 2], [10, 8, 6, 4]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-60.1, -5.5, 29.4]), DOAs_SOI=np.array([-5.5]), DOAs_SOI_perturb=False, SNRs=np.array([[-2, -10, -2], [0, -8, 0], [2, -6, 2], [4, -4, 4], [6, -2, 6], [8, 0, 8], [10, 2, 10], [12, 4, 12], [14, 6, 14], [16, 8, 16], [18, 10, 18]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-69.5, -45.2, -30.1]), DOAs_SOI=np.array([-45.2]), DOAs_SOI_perturb=True, SNRs=np.array([[-14, -10, -15], [-12, -8, -13], [-10, -6, -11], [-8, -4, -9], [-6, -2, -7], [-4, 0, -5], [-2, 2, -3], [0, 4, -1], [2, 6, 1], [4, 8, 3], [6, 10, 5]]), MC_trials=1)

        #tester.compare_metrics(mode="m", DOAs=np.array([-44.4, -19.8, 25.1]), DOAs_SOI=None, DOAs_SOI_perturb=False, SNRs=np.array([[-10, -6, -8], [-8, -4, -6], [-6, -2, -4], [-4, 0, -2], [-2, 2, 0], [0, 4, 2], [2, 6, 4], [4, 8, 6], [6, 10, 8], [8, 12, 10], [10, 14, 12]]), MC_trials=500)
        tester.compare_metrics(mode="m", DOAs=np.array([-4.2, 14.4]), DOAs_SOI=None, DOAs_SOI_perturb=False, SNRs=np.array([[-10, -7], [-8, -5], [-6, -3], [-4, -1], [-2, 1], [0, 3], [2, 5], [4, 7], [6, 9], [8, 11], [10, 13]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-48.3, -20.2, 0.2, 29.5, 60.1]), DOAs_SOI=None, DOAs_SOI_perturb=False, SNRs=np.array([[-6, -9, -8, -7, -6], [-4, -7, -6, -5, -4], [-2, -5, -4, -3, -2], [0, -3, -2, -1, 0], [2, -1, 0, 1, 2], [4, 1, 2, 3, 4], [6, 3, 4, 5, 6], [8, 5, 6, 7, 8], [10, 7, 8, 9, 10], [12, 9, 10, 11, 12], [14, 11, 12, 13, 14]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-50.3, -25.2, 0.2, 25.4, 50.1]), DOAs_SOI=None, DOAs_SOI_perturb=False, SNRs=np.array([[-6, -8, -10, -8, -6], [-4, -6, -10, -6, -4], [-2, -4, -6, -4, -2], [0, -2, -4, -2, 0], [2, 0, -2, 0, 2], [4, 2, 0, 2, 4], [6, 4, 2, 4, 6], [8, 6, 4, 6, 8], [10, 8, 6, 8, 10], [12, 10, 8, 10, 12], [14, 12, 10, 12, 14]]), MC_trials=50)
        #tester.compare_SOI(np.array([-48.3, -20.2, 0.2, 29.5, 60.1]), None, False, np.array([-6, -9, -5, -7, -6]))
        #tester.compare_SOI(np.array([-44.4, -19.8, 25.1]), None, False, np.array([4, 8, 6]))
        #tester.compare_SOI(np.array([-4.2, 14.4]), None, False, np.array([0, 3]))
        #tester.compare_beampattern(np.array([-61.1, -10.3, 6.9, 40]), np.array([6.9]), False, np.array([13, 11, 9, 11]))
        #tester.compare_beampattern(np.array([-45.02, -30.02, -20.02, -3]), np.array([-45.02]), False, np.array([-5, -7, -9, -11]))
        #tester.compare_beampattern(np.array([-60.1, -5.5, 29.4]), np.array([-5.5]), False, np.array([11, 3, 11]))

        #tester.compare_beampattern(np.array([-61.1, -10.3, 6.9, 40]), np.array([6.9]), False, np.array([13, 11, 2, 11]))
        #tester.compare_beampattern(np.array([-60.1, -5.5, 19.4]), np.array([-5.5]), False, np.array([-7, 8, -7]))
        #tester.compare_beampattern(np.array([-45.02, -30.02, -20.02, -3]), np.array([-45.02]), False, np.array([8, -1, 1, 3]))

        #tester.compare_beampattern(np.array([-45.02, -30.02, -20.02, -3]), np.array([-45.02]), True, np.array([-7, -9, -11, -13]))
        #tester.compare_beampattern(np.array([-60.1, -5.5, 29.4]), np.array([-5.5]), True, np.array([9, 1, 9]))

        #tester.compare_metrics(mode="m", DOAs=np.array([-61.1, -10.3, 6.9, 40]), DOAs_SOI=np.array([6.9]), DOAs_SOI_perturb=True, SNRs=np.array([[-6, -8, -10, -8], [-4, -6, -8, -6], [-2, -4, -6, -4], [0, -2, -4, -2], [2, 0, -2, 0], [4, 2, 0, 2], [6, 4, 2, 4], [8, 6, 4, 6], [10, 8, 6, 8], [12, 10, 8, 10], [14, 12, 10, 12]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-45.02, -30.02, -20.02, -3]), DOAs_SOI=np.array([-45.02]), DOAs_SOI_perturb=True, SNRs=np.array([[-10, -12, -14, -16], [-8, -10, -12, -14], [-6, -8, -10, -12], [-4, -6, -8, -10], [-2, -4, -6, -8], [0, -2, -4, -6], [2, 0, -2, -4], [4, 2, 0, -2], [6, 4, 2, 0], [8, 6, 4, 2], [10, 8, 6, 4]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-60.1, -5.5, 29.4]), DOAs_SOI=np.array([-5.5]), DOAs_SOI_perturb=True, SNRs=np.array([[-2, -10, -2], [0, -8, 0], [2, -6, 2], [4, -4, 4], [6, -2, 6], [8, 0, 8], [10, 2, 10], [12, 4, 12], [14, 6, 14], [16, 8, 16], [18, 10, 18]]), MC_trials=500)

        #tester.compare_metrics(mode="m", DOAs=np.array([-61.1, -10.3, 6.9, 40]), DOAs_SOI=np.array([6.9]), DOAs_SOI_perturb=False, SNRs=np.array([[13, 11, 9, 11]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-45.02, -30.02, -20.02, -3]), DOAs_SOI=np.array([-45.02]), DOAs_SOI_perturb=True, SNRs=np.array([[-5, -7, -9, -11]]), MC_trials=500)
        #tester.compare_metrics(mode="m", DOAs=np.array([-60.1, -5.5, 29.4]), DOAs_SOI=np.array([-5.5]), DOAs_SOI_perturb=True, SNRs=np.array([[11, 3, 11]]), MC_trials=500)
