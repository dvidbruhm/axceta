from tools.tests.test_new_wavefront import compute_all_algos
from filterpy.kalman import KalmanFilter


def main():
    df, filt_df, sel_df = compute_all_algos("data/long_series/beaudry-1-5.csv")
    f = KalmanFilter(dim_x=1, dim_z=1)


if __name__ == "__main__":
    main()
