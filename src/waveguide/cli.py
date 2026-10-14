"""Reproducible forward-only CLI; CSV is explicitly synthetic."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from . import Geometry, solve, cutoff_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width",required=True,type=float,help="y width in metres")
    p.add_argument("--height",required=True,type=float,help="z height in metres")
    p.add_argument("--epsilon-real",required=True,type=float)
    p.add_argument("--epsilon-imag",type=float,default=0)
    p.add_argument("--thickness",type=float,default=.00155)
    p.add_argument("--spacing",type=float,default=.032)
    p.add_argument("--spacing-convention",choices=["surface","center"],default="surface")
    p.add_argument("--left-margin",type=float,default=.02)
    p.add_argument("--right-margin",type=float,default=.02)
    p.add_argument("--f-min-ghz",type=float,default=16)
    p.add_argument("--f-max-ghz",type=float,default=18)
    p.add_argument("--points",type=int,default=201)
    p.add_argument("--output",type=Path,default=Path("examples/synthetic_forward.csv"))
    args = p.parse_args()
    if args.points < 2 or args.f_min_ghz >= args.f_max_ghz:
        p.error("Require points>=2 and f-min-ghz<f-max-ghz")
    g = Geometry(args.width,args.height,args.thickness,args.spacing,args.left_margin,args.right_margin,args.spacing_convention)
    out = solve(g,np.linspace(args.f_min_ghz,args.f_max_ghz,args.points)*1e9,complex(args.epsilon_real,args.epsilon_imag))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_GHz","S11_real","S11_imag","S21_real","S21_imag","R","T","A"])
        writer.writerows(zip(out.frequency/1e9,out.s11.real,out.s11.imag,out.s21.real,out.s21.imag,out.R,out.T,out.A))
    metadata = {"source":"SYNTHETIC illustrative forward simulation; no measured data", "geometry_SI":asdict(g),
                "epsilon_real":args.epsilon_real,"epsilon_imag":args.epsilon_imag,
                "time_convention":"exp(-i omega t)","reference_planes_m":[0,g.length],
                "excited_mode":"TE10", "air_cutoffs_Hz":dict(cutoff_table(g))}
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    print(f"Synthetic CSV: {args.output}; TE10 cutoff {g.cutoff/1e9:.6f} GHz")

if __name__ == "__main__": main()
