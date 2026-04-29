import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tikzplotlib

def log_stats(path, msg):
    """ 
    """
    for hand in logging.root.handlers[:]:
        logging.root.removeHandler(hand)
    
    logging.basicConfig(filename=path, format= '%(message)s', level=logging.INFO, force=True)
    logging.info(msg)

    for hand in logging.root.handlers[:]:
        logging.root.removeHandler(hand)
