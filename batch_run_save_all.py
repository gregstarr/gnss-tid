import subprocess
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
import multiprocessing as mp
from argparse import ArgumentParser

ppy = 'plot_save_all.py'

def do_one(ppy, f, kwg):
    subprocess.call(f"python {ppy} {f} {kwg}", shell=True, timeout=60*60*4)
    # print (ppy, f, kwg)
    return


def main(folder, times, j):
    t0, t1 = parser.parse(times[0]), parser.parse(times[1])
    # print (t0, t1)

    latlim= "-90 90"
    lonlim = "-180 180"
    el_mask = 30
    altkm = 250
    res = 0.15
    filter = "gaussian"
    skip = 30
    avg = 10
    filtersize = "5 7 9"
    filtersigma = "0.75 1 1.25"

    kwargs = f"-latlim {latlim} -lonlim {lonlim} -elmask {el_mask} -altkm {altkm} -res {res} -filter {filter} " + \
            f"-skip {skip}  -avg {avg} -filtersize {filtersize} -filtersigma {filtersigma}" 

    all_dates = np.arange(t0, t1+timedelta(hours=1), timedelta(days=1)).astype('datetime64[s]').astype(datetime)

    args = []
    for i,t in enumerate(all_dates):
        args.append((ppy, f"{folder}{t.strftime('%Y/%m%d/')}", kwargs))

    with mp.Pool(processes=j) as pool:
        pool.starmap(do_one, args)
    # print (f"python {ppy} {f} {kwargs}")
    #subprocess.call(f"python {ppy} {f} {kwargs}", shell=True, timeout=60*60*2)

if __name__ == "__main__":
  p = ArgumentParser()
  p.add_argument('folder', help = 'Root Obs Dirctory')
  p.add_argument('times', help = 'start end times. nargs=2', nargs=2, type=str)
  p.add_argument('-j', help = 'Number of parallel processes?. Default=2', type=int, default=2)

  P = p.parse_args()
  main(P.folder, P.times, P.j)
    
