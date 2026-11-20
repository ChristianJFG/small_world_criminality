"""Command-line reproduction of original thesis notebook figures."""
import argparse,json,platform
from pathlib import Path
import numpy as np
import networkx,scipy,seaborn,pandas,matplotlib
from .experiments import run_sweep
from .plots import generate_plots


def main():
    parser=argparse.ArgumentParser(description='Regenerate the ring/square/triangular thesis figures from original notebook logic')
    parser.add_argument('--topologies',nargs='+',choices=['ring','square','triangular'],default=['ring','square','triangular'])
    parser.add_argument('--preset',choices=['quick','manuscript'],default='quick',help='manuscript = 200 nodes, 105 p values x 10 seeds per network; quick = same observable definitions, smaller sample')
    parser.add_argument('--nodes',type=int,default=200)
    parser.add_argument('--seeds',type=int,default=None)
    parser.add_argument('--output',default='results')
    parser.add_argument('--format',choices=['png','svg','pdf'],default='png')
    parser.add_argument('--plot-only',action='store_true',help='Load saved modes.csv instead of recomputing spectra')
    args=parser.parse_args()
    base=Path(args.output)
    for topology in args.topologies:
        out=base/topology;out.mkdir(parents=True,exist_ok=True)
        if args.plot_only:
            import pandas as pd
            modes=pd.read_csv(out/'modes.csv')
            print(f'Replotting saved data from {out}',flush=True)
        else:
            graphs,modes=run_sweep(topology,preset=args.preset,n=args.nodes,seeds=args.seeds,progress=True)
            graphs.to_csv(out/'graphs.csv',index=False)
            modes.to_csv(out/'modes.csv',index=False)
        files=generate_plots(modes,topology,out,args.format)
        metadata=dict(topology=topology,preset=args.preset,nodes=args.nodes,seeds=(10 if args.seeds is None and args.preset=='manuscript' else (2 if args.seeds is None else args.seeds)),
                      modes=len(modes),sources=['ring.ipynb','square.ipynb','triangular.ipynb'],
                      provenance_note='Not bitwise-identical to 2026 manuscript images: historical raw tables were not supplied.',
                      versions=dict(python=platform.python_version(),numpy=np.__version__,networkx=networkx.__version__,scipy=scipy.__version__,seaborn=seaborn.__version__,pandas=pandas.__version__,matplotlib=matplotlib.__version__))
        if not args.plot_only:
            (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(f'Wrote {len(modes)} mode rows and {len(files)} plots for {topology}',flush=True)

if __name__=='__main__':main()
