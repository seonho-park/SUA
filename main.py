import argparse
import numpy as np
import pandas as pd

from extract import load_input_file
from prob import build_model, solve_model

def main() -> None:
    parser = argparse.ArgumentParser(description='Stochastic Programming')
    parser.add_argument('--mu', type=float, default=0.1, help='ratio of aggregate surplus quantity across all products. ranged in [0.1,0.5]')
    parser.add_argument('--ns', type=int, default=100, help='the number of scenarios (samples)')
    parser.add_argument('--solver', type=str, default='highs', choices=['glpk','highs'], help='solver name') 
    parser.add_argument('--seed', type=int, default=1001, help='random seed number')
    args = vars(parser.parse_args()) # to dictionary
    print(args)
    if args['mu'] < 0.1 or args['mu'] > 0.5:
        raise argparse.ArgumentTypeError(f"argument mu should be in range [0.1, 0.5].")

    # set seed for reproducibility
    np.random.seed(args['seed'])

    # load input file and parse it to have parameters
    product, dist = load_input_file()

    # formulate optimization problem
    model = build_model(product, dist, args)

    # solve model
    model = solve_model(model, args['solver'])

    # write output
    pid = []
    surplus = []
    for i in model.P:
        pid.append(i)
        surplus.append(model.x[i].value)
    output = {
        'Product ID': pid,
        'Surplus': surplus
    }
    output_df = pd.DataFrame.from_dict(output)
    output_df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    main()