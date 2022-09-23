# %%
datasetPath = "./TM-3-2020/data/"

# %%
import sys
import os

sys.path.append( os.path.join( os.getcwd( ), './src/lib' ) )

import dataset as ds

# %%
dataset = ds.load_dataset(datasetPath)

# %%
print(dataset[12345]['instructions'])

