import plotly.plotly as py
import plotly.graph_objs as go
import plotly


def plot_data(x_data,y_data, filename="_plot"):
    all_data = []
    for x,y in zip(x_data, y_data):
        all_data.append(go.Scatter(x=x, y=y))
    try:
        py.plot(all_data, filename=filename)
    except:
        # plotly.exceptions.PlotlyError: Hey there! You've hit one of our API request limits.
        pass

