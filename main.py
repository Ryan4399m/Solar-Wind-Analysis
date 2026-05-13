import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.fft import fft, fftfreq


cols = [
    "year", "day", "hour","pts_lanl", "pts_mit", "pts_isee3","speed_lanl", "speed_mit", "speed_isee3",
    "azimuth_lanl", "azimuth_mit", "azimuth_isee3","density_lanl", "density_mit", "density_isee3",
    "temp_lanl", "temp_mit", "temp_isee3","sigma_v_lanl", "sigma_v_mit", "sigma_v_isee3",
    "sigma_n_lanl", "sigma_n_mit", "sigma_n_isee3","sigma_t_lanl", "sigma_t_mit", "sigma_t_isee3",
]

df = pd.read_fwf("data/plasma_data.lst", header=None, names=cols)
df.replace([999.9, 9999.9, 9999999., 99999.9, 999, 9999, 9999.], np.nan, inplace=True)
df = df[df["day"] > 0].copy()  
df["datetime"] = pd.to_datetime(
    df["year"].astype(int).astype(str) + df["day"].astype(int).astype(str).str.zfill(3),
    format="%Y%j"
) + pd.to_timedelta(df["hour"], unit="h")
df.set_index("datetime", inplace=True)

print(df.shape)
print(df.head())
print(df.isnull().sum())

# used the ISEE3 data only since the others had majority NaN
isee = df[["speed_isee3", "density_isee3", "temp_isee3"]].copy()
isee = isee.dropna()
isee = isee.sort_index()

# Rolling avgs 
isee["speed_24h"] = isee["speed_isee3"].rolling("24h").mean()
isee["density_24h"] = isee["density_isee3"].rolling("24h").mean()

# Time series plot 
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig.suptitle("ISEE-3 Solar Wind Plasma Measurements (1979)", fontsize=14)

axes[0].plot(isee.index, isee["speed_isee3"], alpha=0.4, color="blue", linewidth=0.8, label="Hourly")
axes[0].plot(isee.index, isee["speed_24h"], color="blue", linewidth=1.5, label="24h avg")
axes[0].set_ylabel("Speed (km/s)")
axes[0].legend(fontsize=8)

axes[1].plot(isee.index, isee["density_isee3"], alpha=0.4, color="red", linewidth=0.8, label="Hourly")
axes[1].plot(isee.index, isee["density_24h"], color="red", linewidth=1.5, label="24h avg")
axes[1].set_ylabel("Density (N/cm^3)")
axes[1].legend(fontsize=8)

axes[2].plot(isee.index, isee["temp_isee3"], alpha=0.5, color="green", linewidth=0.8)
axes[2].set_ylabel("Temperature (K)")

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/timeseries.png", dpi=150)
plt.show()

# Distribution hist.
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("ISEE-3 Plasma Parameter Distributions", fontsize=13)

axes[0].hist(isee["speed_isee3"], bins=50, color="blue", edgecolor="white")
axes[0].set_xlabel("Speed (km/s)")
axes[0].set_ylabel("Count")

axes[1].hist(isee["density_isee3"], bins=50, color="red", edgecolor="white")
axes[1].set_xlabel("Density (N/cm^3)")

axes[2].hist(isee["temp_isee3"], bins=50, color="green", edgecolor="white")
axes[2].set_xlabel("Temperature (K)")

plt.tight_layout()
plt.savefig("plots/histograms.png", dpi=150)
plt.show()

# Corr. scatter: speed vs density 
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(isee["speed_isee3"], isee["density_isee3"], alpha=0.3, s=8, color="purple")
ax.set_xlabel("Solar Wind Speed (km/s)")
ax.set_ylabel("Plasma Density (N/cm^3)")
ax.set_title("Speed vs. Density (ISEE-3)")

corr = isee["speed_isee3"].corr(isee["density_isee3"])
ax.annotate(f"Pearson r = {corr:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)

plt.tight_layout()
plt.savefig("plots/correlation.png", dpi=150)
plt.show()



# speed data needed to be evenly spaced, so gaps were interpolated
speed_resampled = isee["speed_isee3"].resample("1h").mean().interpolate()

N = len(speed_resampled)
T = 1.0 

yf = np.abs(fft(speed_resampled.values))
xf = fftfreq(N, T)  

# kept only positive frequencies
pos = xf > 0
xf_days = 1 / (xf[pos] * 24) 
yf_pos = yf[pos]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(xf_days, yf_pos, color="darkorange", linewidth=0.8)
ax.set_xlim(1, 40)
ax.set_xlabel("Period (days)")
ax.set_ylabel("Amplitude")
ax.set_title("Fourier Transform of Solar Wind Speed — ISEE-3 (1979)")


peak_idx = np.argmax(yf_pos[(xf_days >= 1) & (xf_days <= 40)])
peak_period = xf_days[(xf_days >= 1) & (xf_days <= 40)][peak_idx]
ax.axvline(peak_period, color="red", linestyle="--", linewidth=1)
ax.annotate(f"Peak: {peak_period:.1f} days", xy=(peak_period, ax.get_ylim()[1]*0.8),
            xycoords="data", fontsize=10, color="red")

plt.tight_layout()
plt.savefig("plots/fourier.png", dpi=150)
plt.show()