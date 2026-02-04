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

def plot_couplings(couplings):
    out_caps, in_caps = couplings.shape
    
    input_positions = [(i, 0) for i in np.linspace(0, out_caps, in_caps)]
    output_positions = [(i, 3) for i in np.linspace(0, out_caps, out_caps)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for out_idx in range(out_caps):
        for in_idx in range(in_caps):
            strength = couplings[out_idx, in_idx].item()
            color = cm.plasma(strength)
            ax.plot(
                [input_positions[in_idx][0], output_positions[out_idx][0]],
                [0, 3],
                color=color,
                linewidth=2 * strength
            )

    for a, b in input_positions:
        ax.plot(a, b, 'o', color="lightcoral")
    for a, b in output_positions:
        ax.plot(a, b, 'o', color="lightcoral")
    
    plt.show()
    #tikzplotlib.save("couplings.tex")
    #fig.savefig("couplings.pdf", bbox_inches="tight")
    #fig.savefig('couplings.svg', format='svg', dpi=1200)
