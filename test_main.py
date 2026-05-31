import pandas as pd
import numpy as np
import pytest
from main import (
    replace_missing_values,
    filter_valid_days,
    build_datetime_index,
    extract_isee,
    compute_rolling_averages,
    compute_correlation
)

def test_missing_value_replacement():
    df = pd.DataFrame({"speed": [500.0, 999.9, 400.0, 9999., 9999999.]})
    result = replace_missing_values(df)
    assert result["speed"].isna().sum() == 3
    assert result["speed"][0] == 500.0
    assert result["speed"][2] == 400.0

def test_filter_valid_days():
    df = pd.DataFrame({
        "year": [1979, 1979, 1979],
        "day": [0, 1, 2],
        "hour": [0, 1, 2]
    })
    result = filter_valid_days(df)
    assert len(result) == 2
    assert 0 not in result["day"].values

def test_datetime_parsing():
    df = pd.DataFrame({
        "year": [1979, 1979],
        "day": [1, 2],
        "hour": [0, 6]
    })
    result = build_datetime_index(df)
    assert result.index[0] == pd.Timestamp("1979-01-01 00:00:00")
    assert result.index[1] == pd.Timestamp("1979-01-02 06:00:00")

def test_extract_isee_drops_nan():
    dates = pd.date_range("1979-01-01", periods=5, freq="h")
    df = pd.DataFrame({
        "speed_isee3": [500.0, np.nan, 400.0, 600.0, np.nan],
        "density_isee3": [5.0, 3.0, np.nan, 4.0, 2.0],
        "temp_isee3": [100000.0, 200000.0, 150000.0, np.nan, 180000.0],
        "other_col": [1, 2, 3, 4, 5]
    }, index=dates)
    result = extract_isee(df)
    assert result.isna().sum().sum() == 0
    assert "other_col" not in result.columns

def test_rolling_average_columns_exist():
    dates = pd.date_range("1979-01-01", periods=48, freq="h")
    isee = pd.DataFrame({
        "speed_isee3": [400.0] * 48,
        "density_isee3": [5.0] * 48,
        "temp_isee3": [100000.0] * 48
    }, index=dates)
    result = compute_rolling_averages(isee)
    assert "speed_24h" in result.columns
    assert "density_24h" in result.columns

def test_rolling_average_values_reasonable():
    dates = pd.date_range("1979-01-01", periods=48, freq="h")
    isee = pd.DataFrame({
        "speed_isee3": [400.0] * 48,
        "density_isee3": [5.0] * 48,
        "temp_isee3": [100000.0] * 48
    }, index=dates)
    result = compute_rolling_averages(isee)
    assert result["speed_24h"].dropna().mean() == pytest.approx(400.0)

def test_correlation_negative():
    dates = pd.date_range("1979-01-01", periods=100, freq="h")
    speed = np.linspace(700, 300, 100)
    density = np.linspace(1, 20, 100)
    isee = pd.DataFrame({
        "speed_isee3": speed,
        "density_isee3": density,
        "temp_isee3": [100000.0] * 100
    }, index=dates)
    corr = compute_correlation(isee)
    assert corr < 0