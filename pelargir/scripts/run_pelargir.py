import sys, os, shutil
import argparse
import matplotlib.pyplot as plt


...


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='pelargir', usage='%(prog)s [options] rundir',
                                     description='Run Pelargir Global population inference')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    ## LATER -- UPDATE TO INFO NEEDED BY PELARGIR

    parser.add_argument('--nofit', action='store_true', help="Disable spectral fit reconstruction plots.")

    parser.add_argument('--cornersplit', type=str, default=None, help="How to split the corner plots. Default None (one corner plot). Can be 'type' or 'submodel'.")

    parser.add_argument('--cornersmooth', type=float, default=0.75, help="Level of 2D contour smoothing for the corner plots. Default 0.75.")

    parser.add_argument('--plotdatadir', type=str, default=None, help="/path/to/plot_data.pickle; where to save the plot data as a pickle file. Defaults to [out_dir]/plot_data.pickle.")

    parser.add_argument('--cornerfmt', type=str, default=None, help="MPL rcParams dict for formatting the corner plots.")
    parser.add_argument('--cornermaxticks', type=int, default=3, help="Maximum number of ticks for the corner plots.")

    # execute parser
    args = parser.parse_args()
