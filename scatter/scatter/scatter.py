import dash
from dash import dcc, html, no_update
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import numpy as np
import pickle


np.set_printoptions(precision=5, linewidth=200)


with open("combined.pkl", "rb") as file:
    df_all = pickle.load(file)


class Global:
    def __init__(self, df_all):
        self.df_all = df_all
        self.switch_prediction(0)

    def switch_prediction(self, i):
        self.dataset_idx = i
        self.df = self.df_all.iloc[self.dataset_idx * 1500: (self.dataset_idx + 1) * 1500]
        self.last_selected_index = 0
        self.selected_index = 0
        self.actual = np.stack(self.df['actual'].apply(np.array).values)
        self.predicted = np.stack(self.df['predicted'].apply(np.array).values)
        self.marker_size = np.full(self.actual.shape[0], 3)
        self.actual_colors = ['rgba(255,0,0, 0.3)'] * self.actual.shape[0]
        self.predicted_colors = ['rgba(0,0,255, 0.3)'] * self.actual.shape[0]


global_state = Global(df_all)


def gen_plot():
    global global_state
    # Create a trace for the scatter plot
    actual_points = go.Scatter3d(
        x=global_state.actual[:, 0],
        y=global_state.actual[:, 1],
        z=global_state.actual[:, 2],
        name='actual',
        mode='markers',
        marker={'color': global_state.actual_colors,
                'size': global_state.marker_size}
    )
    predicted_points = go.Scatter3d(
        x=global_state.predicted[:, 0],
        y=global_state.predicted[:, 1],
        z=global_state.predicted[:, 2],
        name='predicted',
        mode='markers',
        marker={'color': global_state.predicted_colors,
                'size': global_state.marker_size}
    )

    # Create the layout for the plot
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(
                spikecolor='#1fe5bd',
                spikesides=True,
                spikethickness=6,
            ),
            yaxis=go.layout.scene.YAxis(
                spikecolor='#1fe5bd',
                spikesides=True,
                spikethickness=6,
            ),
            zaxis=go.layout.scene.ZAxis(
                spikecolor='#1fe5bd',
                spikethickness=6,
            ),
        ),
        height=700
    )
    # Create the figure object
    fig = go.Figure(data=[actual_points, predicted_points], layout=layout)
    return fig


fig = gen_plot()


def update_plot():
    global global_state

    global_state.marker_size[global_state.last_selected_index] = 3
    global_state.actual_colors[global_state.last_selected_index] = 'rgba(255,0,0, 0.3)'
    global_state.predicted_colors[global_state.last_selected_index] = 'rgba(0,0,255, 0.3)'

    global_state.marker_size[global_state.selected_index] = 20
    global_state.actual_colors[global_state.selected_index] = 'rgba(255,0,0, 1)'
    global_state.predicted_colors[global_state.selected_index] = 'rgba(0,0,255, 1)'

    fig.data[0].marker.color = global_state.actual_colors
    fig.data[0].marker.size = global_state.marker_size
    fig.data[1].marker.color = global_state.predicted_colors
    fig.data[1].marker.size = global_state.marker_size
    return fig


# Create the Dash app and specify the layout
app = dash.Dash()

app.layout = html.Div([
    html.H1("RFID Localization System"),
    dcc.Dropdown(
        options=[
            {'label': f"Prediction {i + 1}", 'value': i} for i in range(7)
        ],
        value=0
        , style={'width': '200px'}, id="dropdown",
        clearable=False
    ),
    html.Br(),
    html.Button('Previous Coordinates', id='previous-button'),
    html.Button('Next Coordinates', id='next-button'),
    html.Button('Reset', id='reset-button'),
    html.Div(id='data-display'),
    # Graph to display the scatter plot
    dcc.Graph(id='3d-scatter', figure=fig),
    # Display the data as a list of strings
    html.Div(id="dummy")
])


@app.callback(
    Output("data-display", "children"),
    Output("3d-scatter", "figure"),
    Input('previous-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('dropdown', 'value'),
)
def display_hover(n_clicks_add, n_clicks_delete, n_clicks_reset, dropdown):
    global global_state, fig
    global_state.last_selected_index = global_state.selected_index
    if dash.callback_context.triggered[0]['prop_id'] == 'next-button.n_clicks':
        global_state.selected_index = (global_state.selected_index + 1) % len(global_state.df)
    elif dash.callback_context.triggered[0]['prop_id'] == 'previous-button.n_clicks':
        global_state.selected_index = (global_state.selected_index - 1 + len(global_state.df)) % len(global_state.df)
    elif dash.callback_context.triggered[0]['prop_id'] == 'reset-button.n_clicks':
        global_state.selected_index = 0
    else:
        global_state.switch_prediction(dropdown)
        fig = gen_plot()
    return display_info(), update_plot()


def display_info():
    global global_state

    df_row = global_state.df.iloc[global_state.selected_index]

    children = []
    raw = np.array(df_row['raw'])
    actual = np.array(df_row['actual'])
    predicted = np.array(df_row['predicted'])

    children.append(html.Pre(f"Point Number: {global_state.selected_index + 1}"))
    children.append(html.Pre(f"Actual: {actual}", style={"color": "red"}))
    children.append(html.Pre(f"Predicted: {predicted}", style={"color": "blue"}))
    children.append(html.Pre(f"Distance: {np.linalg.norm(actual-predicted)}"))

    rssi_list = []
    phase_list = []
    for i in range(len(raw) // 32):
        rssi = raw[i * 32: i * 32 + 16]
        rssi_list.append(rssi)
        phase = raw[i * 32 + 16: i * 32 + 32]
        phase_list.append(phase)
        children.append(html.Pre(f"{i + 1}. RSSI: {rssi.astype(int)}"))
        children.append(html.Pre(f"{i + 1}. Phase: {phase.astype(float)}"))
    children.append(html.Pre(f"Avg. RSSI: {np.mean(rssi_list)}"))
    children.append(html.Pre(f"Avg. Phase: {np.mean(phase_list)}"))
    return children


if __name__ == '__main__':
    app.run_server(debug=True)
