#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A visualization tool to display all experiments logged from using Sacred 
(https://github.com/IDSIA/sacred) and the MongoDB observer and plots their 
logged metrics. 
"""

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from db import sacred_db

# Set colorscheme to use for all graphs
COLORS = plt.get_cmap("Set1").colors

# Initialize dash app
app = dash.Dash(__name__, static_folder='assets')
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# Grab all the experiment and preprocess them
db = sacred_db()
all_experiments, config_cols = db.get_experiment_list()
all_experiments = pd.DataFrame(all_experiments).astype(str)
all_columns, config_cols = list(all_experiments.columns), list(config_cols)

default_disabled_columns = ["last_line", "name"]
enabled_columns = []
for c in all_columns:
    if c not in default_disabled_columns:
        enabled_columns.append(c)


def to_rgba_str(color, alpha=1.0):
    """
    Convert a color to rgba string that plotly expects.
    """
    rgba = [int(c * 255) for c in to_rgb(color)]
    rgbas = "rgba("+str(rgba[0])+","+str(rgba[1]) + \
        ","+str(rgba[2])+","+str(alpha)+")"
    return rgbas


def merge_experiments(experiments):
    """Merge all the experiments with the same configuration but different seed
    """
    merged_expts = experiments.groupby(config_cols).agg({
        "id": list,
        "seed": list
    })
    merged_expts.columns = ["_ids", "_seeds"]
    merged_expts = merged_expts.reset_index()
    merged_expts = merged_expts.rename(
        index=str, columns={
            "_ids": "id",
            "_seeds": "seed"
        })
    merged_expts["seed"] = merged_expts["seed"].map(
        lambda x: "[" + ",".join(x) + "]")
    merged_expts["id"] = merged_expts["id"].map(
        lambda x: "[" + ",".join(x) + "]")
    return merged_expts


"""
Generate a input box to filter table results
"""
filter_selection = html.Div([
    dcc.Input(
        placeholder="Enter search string here",
        id="experiment_filter",
        type="text")
],
                            className="col-6")
"""
Generate checkboxes to enable/disable columns in the table
"""
column_options = [{"label": c, "value": c} for c in all_columns]
column_dropdown = dcc.Dropdown(
    id="column-selection",
    options=column_options,
    multi=True,
    value=enabled_columns)

column_selection = html.Div([
    html.H5(
        "Enabled Columns: ",
        style={
            "margin-bottom": "3px",
            "margin-top": "10px"
        }), column_dropdown
])
"""
Generate a seed merge checkbox
"""
merge_seeds_layout = html.Div([
    dcc.Checklist(
        id="merge-seeds-checklist",
        options=[{
            'label': 'Merge Results Across Seeds',
            'value': 'merge_seeds'
        }],
        values=["merge_seeds"])
])

filter_layout = [
    html.Header([html.H3(["Filters"], style={"margin-bottom": "15px"})],
                className="major"),
    html.Div([merge_seeds_layout, filter_selection, column_selection],
             style={"padding-top": "0px"})
]
"""
Define the application layout
"""
app.layout = html.Div([
    html.Div([
        html.Section(
            [
                html.Div([
                    html.H2(
                        children=["Experiment List"], style={"color": "#000"})
                ],
                         id="header",
                         style={"padding-top": "30px"}),
                html.Div([
                    dt.DataTable(
                        id='table-filtering',
                        row_selectable='multi',
                        selected_rows=[],
                        style_as_list_view=True,
                        sorting=True,
                        columns=[{
                            "name": i,
                            "id": i
                        } for i in enabled_columns],
                    )
                ],
                         className="table-wrapper",
                         style={
                             "margin-top": "20px",
                             "margin-bottom": "0px"
                         }),
            ] + filter_layout + [html.Div([], id="metrics-wrapper")],
            className="inner")
    ],
             id="main")
],
                      id="wrapper")


@app.callback(
    dash.dependencies.Output('table-filtering', 'data'), [
        dash.dependencies.Input('experiment_filter', 'value'),
        dash.dependencies.Input("merge-seeds-checklist", "values")
    ])
def update_experiment_list(filter_val, merge_seeds_options):
    """Update the experimental list by applying any filters 
    and merging them if the merge_seeds option in enabled
    """
    merge_seeds = True if "merge_seeds" in merge_seeds_options else False

    cur_experiments = all_experiments
    if merge_seeds:
        cur_experiments = merge_experiments(cur_experiments)
    if filter_val is None:
        return cur_experiments.to_dict("rows")

    row_mask = np.zeros(len(cur_experiments), dtype=np.bool)
    for col in cur_experiments.columns:
        row_mask += np.array(
            cur_experiments[col].str.contains(filter_val), dtype=np.bool)
    return cur_experiments[row_mask].to_dict("rows")


select_triggers = [
    dash.dependencies.Input('table-filtering', "derived_virtual_data"),
    dash.dependencies.Input('table-filtering', "derived_virtual_selected_rows")
]


@app.callback(
    dash.dependencies.Output('table-filtering', 'style_data_conditional'),
    select_triggers)
def highlight_selected_experiment(virtual_data, virtual_selected_rows):
    """Highlight a "selected" experiment and conditionally color it
    """
    if virtual_selected_rows is not None and len(virtual_selected_rows) > 0:
        return [{
            "if": {
                "row_index": i
            },
            "border": "1px solid #f56a6a",
            "border-left": "0px",
            "border-right": "0px",
            'color': '#f56a6a',
        } for i in virtual_selected_rows]
    else:
        return []


@app.callback(
    dash.dependencies.Output('metrics-wrapper', 'children'), select_triggers)
def display_selected_metrics(virtual_data, virtual_selected_rows):
    """Display all the metrics for the rows selected.
    The "virtual" tag is because they might not correspond to any internal
    python ordering that's expected.
    """
    all_graph_data = {}
    if virtual_selected_rows is not None and len(virtual_selected_rows) > 0:
        for i, virtual_row_no in enumerate(virtual_selected_rows):
            row = virtual_data[virtual_row_no]
            try:
                row["id"] = int(row["id"])
            except Exception as e:
                row["id"] = ast.literal_eval(row["id"])

            # For every experiment that only differ in seeds, aggregate them.
            merged_graph_data = {}
            for experimentids in row["id"]:
                for metric_data in db.get_metrics(experimentids):
                    _, mname, _, msteps, mtstamps, mvals = metric_data.values()

                    if mname not in merged_graph_data:
                        merged_graph_data[mname] = {
                            "x": msteps,
                            "ys": [mvals],
                            "name": str(row["id"])
                        }
                    else:
                        merged_graph_data[mname]["ys"].append(mvals)

            for mname, mdata in merged_graph_data.items():
                if mname not in all_graph_data:
                    all_graph_data[mname] = []

                y_mean = np.mean(mdata["ys"], axis=0)
                y_std = np.std(mdata["ys"], axis=0).tolist()

                trace = go.Scatter(
                    x=mdata["x"],
                    y=y_mean,
                    name=mdata["name"],
                    mode="lines",
                    fill='tonexty',
                    fillcolor=to_rgba_str(COLORS[i], alpha=0.1),
                    line=dict(color=to_rgba_str(COLORS[i], alpha=1.0)),
                )

                upper = go.Scatter(
                    x=mdata["x"],
                    y=y_mean + y_std,
                    fill='tonexty',
                    fillcolor=to_rgba_str(COLORS[i], alpha=0.1),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(width=0))

                lower = go.Scatter(
                    x=mdata["x"],
                    y=y_mean - y_std,
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(width=0))
                all_graph_data[mname].append(lower)
                all_graph_data[mname].append(trace)
                all_graph_data[mname].append(upper)

        all_graphs = []
        for gname, gdata in all_graph_data.items():
            all_graphs.append(
                dcc.Graph(figure={
                    "data": gdata,
                    "layout": go.Layout(title=gname)
                }))
        return html.Div(all_graphs)
    else:
        return []


@app.callback(
    dash.dependencies.Output('table-filtering', 'columns'),
    [dash.dependencies.Input("column-selection", "value")])
def update_enabled_columns(selected_col_ids):
    """Update the enabled columns for the experiment table. 
    Trigggered when the column selection list is modified
    """
    enabled_columns = []
    for c in all_columns:
        if c in selected_col_ids:
            enabled_columns.append(c)
    column_options = [{"name": c, "id": c} for c in enabled_columns]
    return column_options


if __name__ == '__main__':
    app.run_server(debug=True)
