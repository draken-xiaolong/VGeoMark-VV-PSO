"""Small CPU benchmark; deliberately no GPU-specific implementation change."""
import time,json,platform,os
from pathlib import Path
import numpy as np
from psoqim import EmbedConfig,read_shapefile,embed_dataset,extract_dataset
from psoqim.watermark import load_watermark
base=Path(__file__).resolve().parent;d=read_shapefile(base/'data/Railways.shp');w=load_watermark(base/'data/M.png');runs=[]
for i in range(3):
 c=EmbedConfig(n_jobs=1,preserve_parts=True);t=time.perf_counter();e=embed_dataset(d,w,c);embed=time.perf_counter()-t;t=time.perf_counter();r=extract_dataset(e.dataset,w,c,reference_transform=e.transform);extract=time.perf_counter()-t;runs.append(dict(embed_seconds=embed,extract_seconds=extract,nc=r.nc_value))
print(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,cpu_count=os.cpu_count(),runs=runs),indent=2))
