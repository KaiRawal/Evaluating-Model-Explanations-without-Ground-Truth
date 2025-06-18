import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


reference_w1 = 0.7
reference_w2 = 0.3


sector45 = (5,1)
sector90 = (1,5)
sector135 = (-1,5)
sector180 = (-5,1)
sector225 = (-5,-1)
sector270 = (-1,-5)
sector315 = (1,-5)
sector360 = (5,-1)

sectors = [sector45, sector90, sector135, sector180, sector225, sector270, sector315, sector360]

# Create a list of matplotlib polygons for each sector
polygons = [
    Polygon([(0, 0), (1, 0), (1, 1), (0, 0)], closed=True),
    Polygon([(0, 0), (1, 1), (0, 1), (0, 0)], closed=True),
    Polygon([(0, 0), (0, 1), (-1, 1), (0, 0)], closed=True),
    Polygon([(0, 0), (-1, 1), (-1, 0), (0, 0)], closed=True),
    Polygon([(0, 0), (-1, 0), (-1, -1), (0, 0)], closed=True),
    Polygon([(0, 0), (-1, -1), (0, -1), (0, 0)], closed=True),
    Polygon([(0, 0), (0, -1), (1, -1), (0, 0)], closed=True),
    Polygon([(0, 0), (1, -1), (1, 0), (0, 0)], closed=True),
]



def _get_rank(a,b):
    """return positive number if abs(a) is greater than abs(b), and negative if abs(a) is lesser than abs(b)"""
    # print(f'\t{a=}, \t{b=}, \tRANK={abs(a)-abs(b)}')
    return abs(a)-abs(b)

def _get_sign(value: float) -> int:
    """Helper function to determine the sign of a value."""
    return 1 if value >= 0 else -1

def feature_agreement_one(w1: float, w2: float, r1: float = reference_w1, r2: float = reference_w2):
    """
    Function to perform a specific operation using weights and reference weights.
    
    Parameters:
    w1 (float): Weight 1
    w2 (float): Weight 2
    r1 (float): Reference weight 1
    r2 (float): Reference weight 2
    """
    return 1.0 if _get_sign(_get_rank(w1,w2)) == _get_sign(_get_rank(r1,r2)) else 0.0

def sign_agreement_two(w1: float, w2: float, r1: float = reference_w1, r2: float = reference_w2):
    """
    Function to perform a specific operation using weights and reference weights.
    
    Parameters:
    w1 (float): Weight 1
    w2 (float): Weight 2
    r1 (float): Reference weight 1
    r2 (float): Reference weight 2
    """
    sign_r1 = _get_sign(r1)
    sign_w1 = _get_sign(w1)
    sign_r2 = _get_sign(r2)
    sign_w2 = _get_sign(w2)

    if sign_r1 == sign_w1 and sign_r2 == sign_w2:
        return 1.0
    elif sign_r1 != sign_w1 and sign_r2 != sign_w2:
        return 0.0
    else:
        return 0.5

def sign_rank_agreement_two(w1: float, w2: float, r1: float = reference_w1, r2: float = reference_w2):
    """
    Function to perform a specific operation using weights and reference weights.
    
    Parameters:
    w1 (float): Weight 1
    w2 (float): Weight 2
    r1 (float): Reference weight 1
    r2 (float): Reference weight 2
    """
    sign_r1 = _get_sign(r1)
    sign_w1 = _get_sign(w1)
    sign_r2 = _get_sign(r2)
    sign_w2 = _get_sign(w2)
    
    if sign_r1 == sign_w1 and sign_r2 == sign_w2:
        return 1.0 if _get_sign(_get_rank(w1,w2)) == _get_sign(_get_rank(r1,r2)) else 0.0
    elif sign_r1 != sign_w1 and sign_r2 != sign_w2:
        return 0.0
    else:
        return 0.5 if _get_sign(_get_rank(w1,w2)) == _get_sign(_get_rank(r1,r2)) else 0.0

def plot_sectors_superwrapper():
    """
    Superwrapper function to call plot_sectors_wrapper for feature agreement, sign agreement, and signed rank agreement.
    """
    def plot_sectors_wrapper(agreement_function, legtitle):
        """
        Wrapper function to call plot_sectors with predefined shades and reference point.
        """
        ref = (reference_w1, reference_w2)
        shades = []
        for point in sectors:
            agreement = agreement_function(point[0], point[1], ref[0], ref[1])
            if agreement == 0:
                shades.append('mistyrose')
            elif agreement == 0.5:
                shades.append('indianred')
            elif agreement == 1.0:
                shades.append('darkred')
        plot_sectors(shades=shades, legtitle=legtitle, ref=ref)

    # Call the wrapper for each type of agreement
    plot_sectors_wrapper(feature_agreement_one, 'RankAgreement')
    plot_sectors_wrapper(sign_agreement_two, 'SignAgreement')
    plot_sectors_wrapper(sign_rank_agreement_two, 'SignedRankAgreement')

def plot_sectors(shades, legtitle, ref=(0.3,0.2)):
    """
    Function to plot all the polygons defined in the global variable 'polygons' and label the reference point.
    
    Parameters:
    shades (list of str): List of color shades for each polygon. Defaults to a predefined list of colors.
    ref (tuple of float): Reference point to be plotted and labeled.
    """
    
    fig, ax = plt.subplots(layout='constrained')
    for polygon, shade in zip(polygons, shades):
        ax.add_patch(Polygon(polygon.get_xy(), closed=True, color=shade))
    
    # Plot the reference point as a big X
    ax.plot(ref[0], ref[1], 'X', markersize=19, color='white')  # Plot the reference point as a big red X
    ax.text(ref[0]+0.25, ref[1]-0.25, r'$(\beta_1, \beta_2)$', fontsize=25, fontweight='bold', ha='right', va='bottom', color='white')  # Label the reference point with red text
    
    # Set limits and aspect
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', 'box')
    
    # Draw the axes as solid black lines
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Reduce the frequency of ticks on the axes
    ax.set_xticks(np.arange(-1, 2, 1))
    ax.set_yticks(np.arange(-1, 2, 1))
    
    # Label the axes
    ax.set_xlabel(r'$X_1$ importance: $i_1$', fontsize=22)
    ax.set_ylabel(r'$X_2$ importance: $i_2$', fontsize=22)
    # ax.set_xlabel(r'$i_1$', fontsize=20)
    # ax.set_ylabel(r'$i_2$', fontsize=20)
    
    # Add a legend
    legend_labels = {'darkred': '1', 'indianred': '0.5', 'mistyrose': '0'}
    # unique_shades = set(shades)
    unique_shades = ['mistyrose', 'indianred', 'darkred']# set(shades)
    handles = [plt.Line2D([0], [0], color=shade, lw=4) for shade in unique_shades]
    labels = [legend_labels[shade] for shade in unique_shades]
    
    # Add the extra handle for the red 'x' marker
    handles.append(plt.Line2D([0], [0], marker='X', color='black', linestyle='None', markersize=8))
    labels.append(r'$e^{*}$')
    
    # ax.legend(handles, labels, title=legtitle, loc='upper left', fontsize=15, title_fontsize=17, prop={'weight': 'bold'})
    if legtitle == 'SignedRankAgreement':
        ax.legend(handles, labels, loc='lower right', fontsize=18, title_fontsize=19, prop={'weight': 'bold'})
        ax.legend(handles, labels, loc='lower right', fontsize=17)
    




    # Save the plot as a PNG file with a good dpi
    plt.savefig(f'{legtitle}.pdf', dpi=300)
    return 
def main():
    """
    Main function to execute the program.
    """
    plot_sectors_superwrapper()
    pass

if __name__ == "__main__":
    main()

