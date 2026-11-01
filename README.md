# Solar Energy Forecasting
A machine learning application for predicting the daily energy production of an imaginary photovoltaic (PV) system at FH3 building at FH Upper Austria, Hagenberg.
## Goal
The goal of this project is to build a solar forecasting application while learning more about Machine Learning, API development, Docker, LLM integration and security, and Kubernetes — learning what problems would i face in the future developping using these technologies and how to combine all of them together in a project.

## Project overview

- **Goal:** Predict daily PV energy production in kWh/day.
- **Location:** FH Upper Austria, Softwarepark Hagenberg, Austria.
- **PV system:** 5 kWp, south-facing, 35° tilt, and 14% assumed system losses.
- **Coordinates:** 48.368° N, 14.512° E; elevation 439 m.

This is a simulated PV installation. The historical target is modelled PV output from PVGIS, not measurements from a physical installation at the site.

## How it works

1. Historical hourly PV output from PVGIS-SARAH3 is combined with historical hourly weather data from Open-Meteo ERA5.
3. Baseline methods and machine learning models are trained and evaluated using a chronological split.
4. The selected model is retrained using data through 2022 and saved as `models/model.joblib`.
5. For inference, the application retrieves hourly weather forecasts from Open-Meteo and converts them into the daily features expected by the model.
6. The model predicts daily energy production. Where estimates are available, the API also compares the prediction with Forecast.Solar.

## Data

### Sources

- **PVGIS-SARAH3:** Hourly modelled PV system output used to derive the target variable.
- **Open-Meteo ERA5:** Historical hourly weather data used as model inputs.
- **Open-Meteo Forecast API:** Future weather forecasts used when generating predictions.
- **Forecast.Solar:** An independent forecast estimate used for optional comparison with the model's predictions.

The historical dataset covers 2005–2023 (6,939 calendar days before data-quality filtering).

### Target and input features

The target is daily energy production in **kWh/day**, derived from hourly PVGIS power output.

The model uses 13 daily features:

- Temperature at 2 m (`temperature_2m`)
- Relative humidity (`relative_humidity_2m`)
- Total cloud cover (`cloud_cover`)
- Low-, mid-, and high-level cloud cover (`cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`)
- Wind speed at 10 m (`wind_speed_10m`)
- Shortwave, direct, and diffuse radiation (`shortwave_radiation`, `direct_radiation`, `diffuse_radiation`)
- Global tilted irradiance (`global_tilted_irradiance`)
- Two cyclical day-of-year features (`doy_sin`, `doy_cos`)

Daily temperature, humidity, cloud-cover, and wind features are calculated using hourly averages. Radiation features are aggregated as daily sums and converted from Wh/m² to kWh/m². The day-of-year features represent the annual cycle using sine and cosine values.

### Data preparation

1.The source files use different timestamp formats, so timestamps are converted to UTC datetimes before the datasets are joined. The hourly data is then aggregated to one row per day.

2.Some PVGIS records contained zero output during daylight hours. When the sun elevation was above 3° but the recorded output was zero, the affected day was treated as invalid. This removed 266 days (approximately 3.8% of the original calendar days).

i probably will try to find ways to prepare the data better and improve the model 

## Model selection and validation results

The project first evaluated three simple baseline methods. These provide reference points for judging whether machine learning offers an improvement over straightforward forecasting approaches. Three regression models were then evaluated using the same chronological validation period.

### Baseline methods

1. **Baseline 1 — Persistence:** Assumes that today's energy output will be the same as yesterday's output.
2. **Baseline 2 — Seasonal Average:** Uses a historical 15-day smoothed average of energy output for the specific calendar day. It provides a seasonal reference by comparing a day with nearby observations around the same time of year.
3. **Baseline 3 — Cloud Rule:** Estimates how much energy the panels would produce on a clear day, then reduces it based on cloud cover. It was the best-performing baseline.

### Model comparison

The table shows mean absolute error (MAE) on the 2022 validation period; lower values are better. RMSE is also shown for the machine learning models where it was recorded.

| Model / Baseline | MAE (kWh/day) | RMSE (kWh/day) |
|---|---:|---:|
| Baseline 1: Persistence | 6.52 | — |
| Baseline 2: Seasonal Average | 6.86 | — |
| Baseline 3: Cloud Rule | 4.75 | — |
| Linear Regression | 2.756 | 3.590 |
| Random Forest | 2.594 | 3.466 |
| HistGradientBoosting (selected) | **2.514** | **3.353** |

HistGradientBoosting achieved the lowest MAE among the evaluated methods and was selected for the forecasting application. It was then retrained using the historical data through 2022. The saved model artifact is `models/model.joblib`.

**Important:** These are validation results, not final test results. The 2023 holdout should be evaluated separately before reporting final test performance.

## Forecasting API

The FastAPI application provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns the API status and the loaded model name. |
| `GET` | `/forecast?date=YYYY-MM-DD` | Returns the predicted daily energy for a requested future date within the available forecast window. |

The forecast response includes the predicted energy in kWh/day and, when available, the Forecast.Solar estimate and the absolute difference between the two estimates.

The Forecast.Solar comparison is **not a measure of model accuracy**. It compares two forecasts;
## Run with Docker

The repository includes a `Dockerfile` and `.dockerignore`. The Docker image contains the application, its Python dependencies, and the trained model.

### Build the image locally

Run these commands from the project root:

```bash
docker build -t solar-forecast-api:1.0 .
docker run --name solar-forecast -p 8000:8000 solar-forecast-api:1.0
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs).

### Pull the image from Docker Hub

The published image is available at:

[Docker Hub: mouadboumaraf/solar-forecast-api](https://hub.docker.com/r/mouadboumaraf/solar-forecast-api)

To download and run version `1.0` without building it locally:

```bash
docker pull mouadboumaraf/solar-forecast-api:1.0
docker run --name solar-forecast -p 8000:8000 mouadboumaraf/solar-forecast-api:1.0
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to use the API.

## Project structure

```text
Solar_Forecast/
├── api/
│   └── main.py
├── Forecasting/
│   ├── open_meteo_client.py
│   ├── prepare_forecast_inputs.py
│   ├── solar_energy_predictor.py
│   └── forecast_solar_client.py
├── Training/
│   └── ...
├── models/
│   └── model.joblib
├── data/
├── results/
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Future work

- Explore better data sources and improve data quality.
- Experiment with neural networks to improve solar energy predictions.
- Develop a user interface to visualize forecasts and interact with the application.
- Develop a Small Language Model (SLM)-powered assistant to answer questions about solar forecasts.
- Implement security measures to protect the SLM and the application against potential attacks, such as prompt injection.