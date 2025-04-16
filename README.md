# strawberry price forecasting

### OVERVIEW

For this exercise, I tried to adopt simple appraoches to see if we can forecast prices. The `/src` folder contains the models I tried (`NaiveSeasonal` and `LinearRegression`), some preprocessing methods that were required to handle the data for modelling. These methods are used in a jupyter notebook (`/notebook/exploration.ipynb`). The notebook starts with initial exploration of the data, observing missing values in the price targets. It then tries to create features that might contribute towards price movements. Here I tried to use some prior experience in building tomato fruit weight forecasting models I worked on, to check how aggregated cliamte features contribute towards production and thus prices. 

#### DATA SPLITTING
Since the target contains missing values for all years typically all through late summers and fall, I decided to split the data into 10 segments (each segment ranges from winter months to late summer). This resulted in 10 segments (winter 2013 to summer 2014, winter 2014 to summer 2015, .... till winter 2022 to summer 2023), of which I use the first 8 for training and the next 2 for prediction (from late 2021 to summer 2023)).

#### MODELLING

For both models, I used the `darts` timeseries python library. While it has flexibility issues and I would definitely not use this in production code, I have found this to be a nice package for initial experimentation and modelling.

I evaluate model performances using MAPE.

**Baseline**

I then start with a simple baseline model assumming a seasonality (to deduce seasonality I ignored the missing target values and assumed simple asbtract index ranged peak to trough seasonal pattern). 

**Regression**

I then tried linear regression, I tried to investigate these features for modelling:
1. `Lags`: in target. 
2. `Lags`: in climate features. I typically tried lags of 2, 3 and 4 weeks 
3. `Aggregate weather information`: such as total radiation incident over the past 4 weeks, and total temperature sum over the past 4 weeks (The aggregate features are somewhat arbritrary but the rough idea is that fruits develop via radiation and temperature. Literature suggests that the time it takes for a fruit to develop is correlated with the amount of radiation and temperature sum (also called degree-days) experienced by the fruits from flowering to mature stages).
4. `Static features`: such as week of the year (I converted this to a seasonal sin/cos pattern).

For prediction, I tried an auto-regressive way and tried to predict for every week in the test data with a horizon of 3 weeks. The results do not look great in all honesty. 


#### IF I HAD MORE TIME
I spent a total of about 5 hours on this exercise. This is nearly not enought and if I had more time I would probably;
1. first run a grid search on the lags, and some common hyperparameters. 
2. I would then investigate the features a bit more. 
3. Perhaps I would like to see how the performance looks if we try an ensemble method (regression + boosting for example?). 
4. Clean up the code a bit more, use pipelines, write inference scripts, minimal tests, add logging, log and track trainings and model performance in mflow to better track progress.



## Repository Structure
```
strawberry_price_forecast/
├── src/
│ ├── models.py # Model classes and pipeline definitions
│ ├── process.py # Data preprocessing utilities
├── data/ # Data files (not tracked)
│ ├── senior_ds_test.csv # the given file
├── notebook/ # Jupyter notebook for for plotting and modelling
│ └── exploration.ipynb
├── pyproject.toml # Poetry dependencies
└── poetry.lock # poetry lock file
```


## Setup

### Environment
We use Poetry for dependency management under a `Python 3.11` environment. You can follow the steps below to install Poetry. Alternatively the dependencies are listed in `pyproject.toml` and you can install them in your own environment in your preferred manner.

2. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies (run from root):
```bash
poetry install
```

## Dependency management

Add dependencies if needed
```bash
poetry add package_name
```
