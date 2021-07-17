# %%
import os
# os.chdir(snakemake.params.proj_dir)

# %%
from tables import *
import pandas as pd
import subprocess
from datetime import datetime

h5_ins = snakemake.input
# h5_ins = [
#    "data/interim/Predictions/Model_Preds/AnyLink_{:03}.hdf5".format(x) for x in range(6)]
h5_comp = snakemake.output.h5_comp
# h5_comp = "data/processed/Predictions/AnyLink.hdf5.lz4"
# max_str_len = snakemake.params.max_str_len
ind_size = snakemake.params.ind_size
# ind_size = 203848336
name = snakemake.params.name
# name = "AnyLink"
# %%
filters = Filters(complevel=1, complib='blosc', shuffle=True, least_significant_digit=4)
filters_ind = Filters(complevel=1, complib='blosc', shuffle=True)
out = open_file(h5_comp, "w", title="Data")
group = out.create_group("/", "data")

# %%
print("Merging....")
st = datetime.now()
# inp=h5_ins[0]
for i, inp in enumerate(sorted(h5_ins)):
    print("-- merging {}".format(inp))
    tmp = open_file(inp, "r")
    cols = tmp.root.data.data.description
    if i == 0:
        table = out.create_table(group, name, cols, "Probas {}".format(
            name), expectedrows=ind_size, filters=filters, chunkshape=1000)
    table.append(tmp.root.data.data.read())
    table.flush()
    print(
        "-- done!, elapsed {:.2f} seconds".format((datetime.now() - st).total_seconds()))
    print("Current store info:\n", table)
#table.cols.ind.create_csindex(filters=filters_ind)
out.close()
print("Finished all!, elapsed {:.2f} seconds".format(
    (datetime.now() - st).total_seconds()))

# %% compress
# subprocess.run(["ptrepack", "--sortby", "ind",
#                 "--chunkshape", "auto",
#                 "--propindexes", "--complevel", "6",
#                 "--complib", "blosc:lz4",
#                 h5_comp[:-4] + ":/data/{}".format(name), h5_comp + ":/data/{}".format(name)])
# subprocess.run(['rm', '-f', h5_comp[:-4]])
