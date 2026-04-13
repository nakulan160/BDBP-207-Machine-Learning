import pandas as pd
import numpy as np

def main():
    data = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    set1=data[data["BP"]>80]
    set2=data[data["BP"]<=80]





if __name__ == "__main__":
        main()