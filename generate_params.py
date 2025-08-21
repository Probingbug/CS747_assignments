import numpy as np
import pandas as pd
import argparse

def generate_params(size=340, out_file="cmaes_params.json"):
    params = np.random.uniform(-1, 1, size=size)
    df = pd.DataFrame({'Params': [params.tolist()]})
    df.to_json(out_file)
    print(f"✅ Saved {size}-length params to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=340, help="Size of parameter vector")
    parser.add_argument("--file", type=str, default="cmaes_params.json", help="Output filename")
    args = parser.parse_args()

    generate_params(args.size, args.file)