"""Plotly figure builders.

These are intentionally lightweight static figures. The original app rebuilt
60-frame client-side animations on every Streamlit rerun, which was heavy and
flickery; a clean static chart with hover detail performs far better and reads
just as well. Plotly's own transition easing gives a subtle animated update.
"""
from __future__ import annotations

import plotly.graph_objects as go

_PALETTE = ["#00ffd5", "#00a9ff", "#7c5cff", "#ff7ad9", "#ffd166",
            "#06d6a0", "#ef476f", "#118ab2"]


def _layout(title: str, y_max: float | None = None, shapes=None) -> dict:
    layout = dict(
        template="plotly_dark",
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=46, b=10),
        xaxis=dict(tickangle=20),
        transition=dict(duration=350, easing="cubic-in-out"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if y_max is not None:
        layout["yaxis"] = dict(range=[0, y_max])
    if shapes:
        layout["shapes"] = shapes
    return layout


def _ref_line(n: int, y: float, color: str, label: str) -> dict:
    return dict(
        type="line", x0=-0.5, x1=n - 0.5, y0=y, y1=y,
        line=dict(dash="dash", color=color, width=1.5),
        label=dict(text=label, textposition="end"),
    )


def bar(labels, values, title, y_max=None, pass_line=None, target_line=None):
    shapes = []
    if pass_line is not None:
        shapes.append(_ref_line(len(labels), pass_line, "#ef476f", f"Pass {pass_line:g}"))
    if target_line is not None:
        shapes.append(_ref_line(len(labels), target_line, "#ffd166", f"Target {target_line:g}"))
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker=dict(color=list(values), colorscale="Tealgrn", line=dict(color="#22d3ee", width=1)),
        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**_layout(title, y_max=y_max, shapes=shapes))
    return fig


def pie(labels, values, title):
    if float(sum(values)) <= 0:
        return go.Figure().update_layout(
            **_layout(title),
            annotations=[dict(text="No data yet", x=0.5, y=0.5, showarrow=False)],
        )
    fig = go.Figure(go.Pie(
        labels=list(labels), values=list(values), hole=0.35,
        marker=dict(colors=_PALETTE), textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:.1f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(**_layout(title))
    return fig


def line(labels, values, title, y_max=None):
    fig = go.Figure(go.Scatter(
        x=list(labels), y=list(values), mode="lines+markers",
        line=dict(color="#00ffd5", width=2.5), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(0,255,213,0.08)",
        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**_layout(title, y_max=y_max))
    return fig


def time_series(x, y, title, y_max=None):
    fig = go.Figure(go.Scatter(
        x=list(x), y=list(y), mode="lines+markers",
        line=dict(color="#00a9ff", width=2.5), marker=dict(size=8),
        hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
    ))
    layout = _layout(title, y_max=y_max)
    layout["xaxis"] = dict()
    fig.update_layout(**layout)
    return fig
