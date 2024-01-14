import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from streamlit_plotly_events import plotly_events
from lmfit.models import GaussianModel, ConstantModel
from lmfit import Model


uploaded_file = st.sidebar.file_uploader("Choose a file")
model_selection = st.sidebar.selectbox(
    "Select Model", ["Linear", "2D-Polynomial"], key="model_selection"
)

ylabel = st.sidebar.text_input("y-label", "y")
xlabel = st.sidebar.text_input("x-label", "x")
title_plot = st.sidebar.text_input("title", "title")
xlabel_size = st.sidebar.slider("x-label size", 0, 100, 20)
ylabel_size = st.sidebar.slider("y-label size", 0, 100, 20)

fit_button = st.sidebar.button("fit")

if uploaded_file is not None:
    x, y = np.loadtxt(uploaded_file, delimiter=",", unpack=True)


def linearModel(x, a0, a1):
    return a0 + a1 * x


def polynomialModel(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x**2


def plotter():
    fig1 = px.scatter(x=x, y=y)
    fig1.update_layout(title_text=f"title_plot", title_x=0.5)
    fig2 = px.line(x=x, y=results.best_fit)
    fig2.data[0].line.color = "red"
    fig1.data[0].error_y.array = results.residual
    fig1.data[0].error_y.visible = True
    fig1.data[0].error_y.type = "data"
    fig1.data[0].error_y.color = "lightblue"
    fig1.update_layout(
        yaxis=dict(title=ylabel, titlefont_size=ylabel_size),
        xaxis=dict(title=xlabel, titlefont_size=xlabel_size),
    )
    fig1.add_trace(fig2.data[0])
    st.plotly_chart(fig1)


if model_selection == "Linear":
    if fit_button:
        mod = Model(linearModel)
        pars = mod.make_params(a0=0, a1=1)
        results = mod.fit(y, pars, x=x)
        plotter()
        offset = results.best_values["a0"]
        slope = results.best_values["a1"]
        offset_error = results.params["a0"].stderr
        slope_error = results.params["a1"].stderr
        st.write(
            rf"offset: ${offset:.2f},\pm {offset_error:.2f}$, slope: ${slope:.2f},\pm {slope_error:.2f}$ \, $\chi^{2}: {results.chisqr:.2f}$",
            r"$\chi_{\nu}^{2}$:" f"{results.redchi:.2f}",
        )
elif model_selection == "2D-Polynomial":
    if fit_button:
        mod = Model(polynomialModel)
        pars = mod.make_params(a0=0, a1=1, a2=1)
        results = mod.fit(y, pars, x=x)
        plotter()
        a0_error = results.params["a0"].stderr
        a1_error = results.params["a1"].stderr
        a2_error = results.params["a2"].stderr
        st.write(
            f"offset: {results.best_values['a0']:.2f},$\pm {a0_error:.2f}$, slope: {results.best_values['a1']:.2f}, $\pm {a1_error:.2f}$, quadratic: {results.best_values['a2']:.2f} $\pm {a2_error:.2f}$"
        )
        st.write(
            rf"$\chi^{2}: {results.chisqr:.2f}$",
            r"$\chi_{\nu}^{2}$: " f"{results.redchi:.2f}",
        )
