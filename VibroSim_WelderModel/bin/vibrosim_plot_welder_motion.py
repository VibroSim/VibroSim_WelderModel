import os
import os.path
import sys
import collections
import tempfile

try:
    # py2.x
    from urllib import pathname2url
    from urllib import url2pathname
    from urllib import quote
    from urllib import unquote
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    from urllib.request import url2pathname
    from urllib.parse import quote
    from urllib.parse import unquote
    pass

import pandas as pd

from matplotlib import pyplot as pl

from limatix.dc_value import numericunitsvalue as numericunitsv
from limatix.dc_value import hrefvalue as hrefv

from VibroSim_WelderModel import contact_model

def main(args=None):
    if args is None:
        args=sys.argv
        pass
    
    if len(args) < 2:
        print("Usage: %s <motionfile.csv.bz2>" % (args[0]))
        sys.exit(0)
        pass

    motionfile = args[1]

    motiontable = pd.read_csv(motionfile,index_col=0)
    

    # Generate plots
    plotdict = contact_model.plot_contact(motiontable)
    
    ret = collections.OrderedDict()

    
    
    # Save plots to disk
    tempdir = tempfile.gettempdir()
    print("Saving plots in %s" % tempdir)
    for plotdescr in plotdict:
        pl.figure(plotdict[plotdescr].number)
        savename = os.path.join(tempdir,"welderplot_%s.png" % (plotdescr))
        print("Saving figure %d as %s" % (plotdict[plotdescr].number,savename))
        pl.savefig(savename,dpi=300)
        pass

    print("Close all plot windows and/or call quit() function to exit")
    
    pl.show()
    
    pass
