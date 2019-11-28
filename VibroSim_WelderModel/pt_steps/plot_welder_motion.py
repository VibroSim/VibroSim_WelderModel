import os
import os.path
import sys

from dc_value import numericunitsvalue as numericunitsv
from dc_value import hrefvalue as hrefv

from VibroSim_WelderModel import contact_model

def run(dc_dest_href,
        dc_measident_str,
        dc_motion_href):

    motiontable = pd.read_csv(dc_motion_href.getpath(),index_col=0)
    

    # Generate plots
    plotdict = contact_model.plot_contact(motiontable)
    
    ret = collections.OrderedDict()

    # Save plots to disk and add to return dictionary
    for plotdescr in plotdict:
        pl.figure(plotdict[plotdescr].number)
        plot_href = hrefv(quote("%s_%s.png" % (dc_measident_str,plotdescr)),dc_dest_href)        
        pl.savefig(plot_href.getpath(),dpi=300)
        ret["dc:%s_plot" % (plotdescr)] = plot_href
        pass
    
    return ret
