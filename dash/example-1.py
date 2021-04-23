# %%
import dash
import numpy as np
import pandas as pd

import plotly.express as px

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# %%
np.random.randint(1, 4, 3)

# %%
style = {'width': '100%', 'display': 'inline-block'}

app = dash.Dash(__name__)

marks1 = dict()
for x in range(5, 20):
    marks1[x] = f'{x}'


marks2 = dict()
for x in [10, 20, 30, 40]:
    marks2[x] = f'{x}'

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Slider(
                    id='slider1',
                    value=10,
                    min=5,
                    max=20,
                    marks=marks1,
                )
            ],
            style=style
        ),
        html.Div(
            [
                dcc.Graph(id='graph1')
            ],
            style=style),
        html.Div(
            [
                dcc.Slider(
                    id='slider2',
                    value=20,
                    min=10,
                    max=40,
                    marks=marks2,
                )
            ],
            style=style
        ),
    ]
)

# %%


class MyFigure(object):
    def __init__(self, num=10):
        self.generate_fig(num=num)

    def generate_fig(self, num):
        df = pd.DataFrame()
        df['id'] = range(num)
        df['size'] = df['id'].map(lambda e: np.random.randint(5, 10))
        df['x'] = df['id'].map(lambda e: np.random.randint(5, 10))
        df['y'] = df['id'].map(lambda e: np.random.randint(5, 10))
        df['color'] = df['id'].map(lambda e: np.random.randint(0, 3))
        fig = px.scatter(df, x='x', y='y', color='color', size='size')
        print(f'Generate figure with num="{num}"')
        self.fig = fig


myFig = MyFigure(10)

# %%

# -----------------------------------------------------------------


@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [
        Input(component_id='slider1', component_property='value'),
        Input(component_id='slider2', component_property='value'),
    ]
)
def update_graph1(num1, num2):
    triggered = dash.callback_context.triggered[0]

    if triggered['prop_id'].startswith('slider1'):
        print(f'Slide 1 = {num1}')
        myFig.generate_fig(num1)

    if triggered['prop_id'].startswith('slider2'):
        print(f'Slide 2 = {num2}')
        fig = myFig.fig
        fig['data'][0]['marker']['size'][0] = num2

    return myFig.fig


# %%
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
