import os
import os.path
import sys
import collections

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

def run(dc_dest_href,
        dc_measident_str,
        dc_motion_href,
        dc_exc_t0_numericunits):

    motiontable = pd.read_csv(dc_motion_href.getpath(),index_col=0)
    

    # Generate plots
    plotdict = contact_model.plot_contact(motiontable,dc_exc_t0_numericunits.value("s"))
    
    ret = collections.OrderedDict()

    # Save plots to disk and add to return dictionary
    for plotdescr in plotdict:
        pl.figure(plotdict[plotdescr].number)
        plot_href = hrefv(quote("%s_%s.png" % (dc_measident_str,plotdescr)),dc_dest_href)        
        pl.savefig(plot_href.getpath(),dpi=300)
        ret["dc:%s_plot" % (plotdescr)] = plot_href
        pass
    
    return ret
