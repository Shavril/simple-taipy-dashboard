# simple-taipy-dashboard
Simple data science dashboard using Taipy.

## Acknowledgement

I used this [Youtube tutorial from MariyaSha](https://www.youtube.com/watch?v=hxYIpH94u20) to create this dashboard. 

## Introduction

A stock value dashboard using `Taipy`, `Plotly`, and [a dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks).
It will dynamically filter data, display graphs, and handle user inputs—all from scratch.

## Set up

1. Clone the repository.

2. Put [this dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks) into folder `/data/`:
    ```
    /data/sp500_companies.csv
    /data/sp500_index.csv
    /data/sp500_stocks.csv
    ```

3. This project uses `Poetry` for package management.  
   Complete package versions are specified in `pyproject.toml` file.
   WHen you have poetry installed, go to the directory with .toml file,
   and run `poetry install`.

4. Run `poetry run python main.py`. 
   It will serve the web application locally on [http://localhost:5000](http://localhost:5000).
