import logging

def log_stats(path, msg):
    """ 
    """
    for hand in logging.root.handlers[:]:
        logging.root.removeHandler(hand)
    
    logging.basicConfig(filename=path, format= '%(message)s', level=logging.INFO, force=True)
    logging.info(msg)

    for hand in logging.root.handlers[:]:
        logging.root.removeHandler(hand)