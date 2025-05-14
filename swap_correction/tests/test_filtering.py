import numpy as np
import pandas as pd
from swap_correction.tracking.filtering import filters
from swap_correction import metrics

def make_full_df(n=10):
    # All columns from POSDICT
    cols = [col for pair in metrics.POSDICT.values() for col in pair]
    data = {col: np.arange(n) for col in cols}
    return pd.DataFrame(data)

def test_filter_sgolay():
    df = make_full_df()
    filtered = filters.filter_sgolay(df, window=5, order=2)
    assert isinstance(filtered, pd.DataFrame)
    assert filtered.shape == df.shape 