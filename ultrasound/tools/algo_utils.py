import pandas as pd
import tools.utils as utils
import tools.ultrasound_algos as ua


def run_data_on_multiple_batch(file, funcs: dict):
    df = pd.read_csv(file)
    df["raw_data"] = df.apply(lambda row: utils.str_raw_data_to_list(row["raw_data"]), axis=1)
    df["new_wf"] = df.apply(
        lambda row: ua.wavefront_empty_and_full_detection(
            row["raw_data"], 0.5, row["pulseCount"], row["samplingFrequency"], row["maxBinIndex"] * (row["samplingFrequency"] / 1e6)
        ),
        axis=1,
    )
    df["new_quality"] = df.apply(lambda row: ua.signal_quality(row["raw_data"])["c2"], axis=1)

    # df[df.groupby(['id'])['year'].transform(max) == df['year']]
