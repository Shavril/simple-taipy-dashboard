# Stock Data Dashboard Application
# Author: Michaela Honkova
# Tutorial used: https://www.youtube.com/watch?v=hxYIpH94u20 by Mariya Sha

from datetime import datetime
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from tf_keras.models import Sequential
from tf_keras.layers import Dense

import taipy as tp
import taipy.gui.builder as tgb
from taipy.gui import Icon
from taipy import Config

######################
#  Global Variables  #
######################

country="United States"
company=[]

lin_pred = 0
knn_pred = 0
rnn_pred = 0

graph_data = None
figure = None

stock_data = pd.read_csv("data/sp500_stocks.csv")
company_data = pd.read_csv("data/sp500_companies.csv")

country_names = company_data["Country"].unique().tolist()
country_names = [(name, Icon(f"images/flags/{name}.png", name)) for name in country_names]
company_names = company_data[["Symbol", "Shortname"]].sort_values("Shortname").values.tolist()

dates = [
    stock_data["Date"].min(),
    stock_data["Date"].max(),
]

############################
#  Graphic User Interface  #
############################

# create page
with tgb.Page() as page:

    # create horizontal group of elements
    with tgb.part("text-center"):
        tgb.image("images/icons/logo.png", width="10vw")
        tgb.text("# S&P 500 Stock Value Over Time", mode="md")

        # date range selector
        tgb.date_range(
            "{dates}",
            label_start="Start Date",
            label_end="End Date",
        )

        # country selector (20% width) and
        # company selector (80% width)
        with tgb.layout("20 80"):
            tgb.selector(
                label="country",
                class_name="fullwidth",
                value="{country}",
                lov="{country_names}",
                dropdown=True,
                value_by_id=True,
            )
            tgb.selector(
                label="company",
                class_name="fullwidth",
                value="{company}",
                lov="{company_names}",
                dropdown=True,
                value_by_id=True,
                multiple=True,
            )

        # chart
        tgb.chart(figure="{figure}")

        # vertical group of 8 elements
        with tgb.part("text-left"):
            with tgb.layout("4 72 4 4 4 4 4 4"):

                # company
                tgb.image("images/icons/id-card.png", width="3vw")
                tgb.text(
                    # display company symbol (shortcut) and its name
                    value="{company[-1]} | {company_data['Shortname'][company_data['Symbol'] == company[-1]].values[0]}",
                    mode="md",
                )

                # linear regression
                tgb.image("images/icons/lin.png", width="3vw")
                tgb.text("{lin_pred}", mode="md")

                # K-nearest neighbors
                tgb.image("images/icons/knn.png", width="3vw")
                tgb.text("{knn_pred}", mode="md")

                # recurrent neural network
                tgb.image("images/icons/rnn.png", width="3vw")
                tgb.text("{rnn_pred}", mode="md")

###############
#  Functions  #
###############

def build_company_names(
        country: str,
) -> list[str]:
    """
    Filter companies by their country of origin.

    :param country: name of the country
    :return: list of company names
    """
    company_names = company_data[["Symbol", "Shortname"]][
        company_data["Country"] == country
    ].sort_values("Shortname").values.tolist()
    return company_names


def build_graph_data(
        dates: list[str|datetime],
        company: list[str],
) -> pd.DataFrame:
    """
    Filter the stock data by dates and companies.

    :param dates: list with start and end date
    :param company: list of company symbols
    :return: filtered stock data frame
    """
    temp_data = stock_data[["Date", "Adj Close", "Symbol"]][
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))
    ]

    # blank dataframe
    graph_data = pd.DataFrame()

    # add column 'Date'
    graph_data["Date"] = temp_data["Date"].unique()

    # add column 'Adj Close' for each company, the column will be named after the 'company'
    for i in company:
        graph_data[i] = temp_data["Adj Close"][
            temp_data["Symbol"] == i
        ].values

    return graph_data


def display_graph(
        graph_data: pd.DataFrame,
) -> go.Figure():
    """
    Draw figure with stock value lines.

    :param graph_data: data to be drawn
    :return: drawn figure
    """
    figure = go.Figure()
    symbols = graph_data.columns[1:]

    for i in symbols:
        figure.add_trace(go.Scatter(
            x=graph_data["Date"],
            y=graph_data[i],
            name=i,
            showlegend=True,
        ))
    figure.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Value",
    )

    return figure


def split_data(
        stock_data: pd.DataFrame,
        dates: list[str|datetime],
        symbol: str,
        normalize: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Extract data for training and prediction.

    :param stock_data: the stock data frame
    :param dates: start and end date
    :param symbol: company symbol (shortcut)
    :param normalize: if we need to normalize the data (we do for rnn model)
    :return: three ndarrays with data
    """
    # filter the dataframe
    temp_data = stock_data[
        (stock_data["Symbol"] == symbol) &
        (stock_data["Date"] > str(dates[0])) &
        (stock_data["Date"] < str(dates[1]))
    ].drop(["Date", "Symbol"], axis=1)

    # last row of the features, for which we want to estimate target
    eval_features = temp_data.values[-1]

    # its shape must match shape of features for lin_reg
    eval_features = eval_features.reshape(1, -1)

    # all other previous rows of features
    features = temp_data.values[:-1]

    # the closing value of a day is target of previous day:
    # we shift the closing values by -1 row to get targets
    # we slice off last row because that doesnt have value yet
    targets = temp_data["Adj Close"].shift(-1).values[:-1]

    # normalize the data
    if normalize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        features -= mean
        features /= std
        eval_features -= mean
        eval_features /= std

    return features, targets, eval_features


def get_lin(
        dates: list[str|datetime],
        company: list[str],
) -> float:
    """
    Get prediction using linear regression algorithm.

    :param dates: start and end dates
    :param company: list of company symbols
    :return: float prediction for next day
    """
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    lin_model.fit(x, y)
    lin_pred = lin_model.predict(eval_x)
    return round(lin_pred[0], 3)


def get_knn(
        dates: list[str|datetime],
        company: list[str],
) -> float:
    """
    Get prediction using K-Nearest Neighbors algorithm.

    :param dates: start and end dates
    :param company: list of company symbols
    :return: float prediction for next day
    """
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    knn_model.fit(x, y)
    knn_pred = knn_model.predict(eval_x)
    return round(knn_pred[0], 3)


def get_rnn(
        dates: list[str|datetime],
        company: list[str],
) -> float:
    """
    Get prediction using Recurrent Neural Network algorithm.

    :param dates: start and end dates
    :param company: list of company symbols
    :return: float prediction for next day
    """
    x, y, eval_x = split_data(stock_data, dates, company[-1], normalize=True)
    rnn_model.fit(x, y, batch_size=32, epochs=10, verbose=0)
    rnn_pred = rnn_model.predict(eval_x)
    return round(float(rnn_pred[0][0]), 3)


#############
#  Backend  #
#############

# configure data nodes
country_cfg = Config.configure_data_node(
    id="country",
)
company_names_cfg = Config.configure_data_node(
    id="company_names",
)
dates_cfg = Config.configure_data_node(
    id="dates",
)
company_cfg = Config.configure_data_node(
    id="company",
)
graph_data_cfg = Config.configure_data_node(
    id="graph_data",
)
lin_pred_cfg = Config.configure_data_node(
    id="lin_pred",
)
knn_pred_cfg = Config.configure_data_node(
    id="knn_pred",
)
rnn_pred_cfg = Config.configure_data_node(
    id="rnn_pred",
)

# configure tasks
build_company_names_cfg = Config.configure_task(
    input=country_cfg,
    output=company_names_cfg,
    function=build_company_names,
    id="build_company_names",
    skippable=True,
)

build_graph_data_cfg = Config.configure_task(
    input=[dates_cfg, company_cfg],
    output=graph_data_cfg,
    function=build_graph_data,
    id="build_graph_data",
    skippable=True,
)

get_lin_cfg = Config.configure_task(
    input=[dates_cfg, company_cfg],
    output=lin_pred_cfg,
    function=get_lin,
    id="get_lin",
    skippable=True,
)

get_knn_cfg = Config.configure_task(
    input=[dates_cfg, company_cfg],
    output=knn_pred_cfg,
    function=get_knn,
    id="get_knn",
    skippable=True,
)

get_rnn_cfg = Config.configure_task(
    input=[dates_cfg, company_cfg],
    output=rnn_pred_cfg,
    function=get_rnn,
    id="get_rnn",
    skippable=True,
)

# configure scenario
scenario_cfg = Config.configure_scenario(
    task_configs=[
        build_company_names_cfg,
        build_graph_data_cfg,
        get_lin_cfg,
        get_knn_cfg,
        get_rnn_cfg,
    ],
    id="scenario",
)


def on_init(
        state,
) -> None:
    """
    Built-in Taipy function that runs
    once when the application first loads.

    :param state:
    :return:
    """
    # input
    state.scenario.country.write(state.country)
    state.scenario.dates.write(state.dates)
    state.scenario.company.write(state.company)

    # process
    state.scenario.submit(wait=True)

    # output
    state.graph_data = state.scenario.graph_data.read()
    state.company_names = state.scenario.company_names.read()
    state.lin_pred = state.scenario.lin_pred.read()
    state.knn_pred = state.scenario.knn_pred.read()
    state.rnn_pred = state.scenario.rnn_pred.read()


# Fetch selection every time gui value is modified
def on_change(
        state,
        name: str,
        value,
) -> None:
    """
    Built-in Taipy function that runs
    every time a GUI variable is changed by user.

    :param state:
    :param name:
    :param value:
    :return:
    """
    if name == "country":
        # update scenario with new country value
        print(name, "was modified to", value)
        state.scenario.country.write(state.country)
        state.scenario.submit(wait=True)
        state.company_names = state.scenario.company_names.read()

    if name == "company" or name == "dates":
        # update scenario with new company or dates selection
        print(name, "was modified to", value)
        state.scenario.dates.write(state.dates)
        state.scenario.company.write(state.company)
        state.scenario.submit(wait=True)
        state.graph_data = state.scenario.graph_data.read()
        state.lin_pred = state.scenario.lin_pred.read()
        state.knn_pred = state.scenario.knn_pred.read()
        state.rnn_pred = state.scenario.rnn_pred.read()

    if name == "graph_data":
        # display updated graph data
        state.figure = display_graph(state.graph_data)


def build_RNN(
        n_features: int,
) -> Sequential():
    """
    Create a Recurrent Neural Network,

    :param n_features: number of features within x and eval_x
    :return: RNN Tensorflow model
    """
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(n_features, )))
    model.add(Dense(units=64, activation='relu',))
    model.add(Dense(units=1, activation='linear',))
    model.compile(optimizer = 'rmsprop', loss='mse', metrics=['mae'],)
    return model


if __name__ == "__main__":

    # create machine learning models
    lin_model = LinearRegression()
    knn_model = KNeighborsRegressor()
    rnn_model = build_RNN(6)  # 6 features (columns) in model

    # run Taipy orchestrator to manage scenario
    tp.Orchestrator().run()

    # intialize scenario
    scenario = tp.create_scenario(scenario_cfg)

    # initialize GUI and display page
    gui = tp.Gui(page)

    # run application
    gui.run(
        title = "Data Science Dashboard",
        # automatically reload app when main.py is saved
        # use_reloader=True,
    )
