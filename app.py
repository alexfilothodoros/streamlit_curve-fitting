import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel
from lmfit import Model

plt.switch_backend("Qt5Agg")

fit_button = st.sidebar.button("fit")
# Generate random data
x = np.linspace(-5, 5, 100)
y = np.exp(-(x ** 2))

# Create figure
fig = px.line(x=x, y=y, title="Gaussian Curve2")

# Show figure in Streamlit
selected_points = plotly_events(fig, key="my_event")
if selected_points:
    xclick = selected_points[0]["x"]
    yclick = selected_points[0]["y"]


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / wid)


if fit_button:
    mod = Model(gaussian)
    pars = mod.make_params(amp=yclick, cen=xclick, wid=1)
    results = mod.fit(y, pars, x=x)
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data")
    ax.plot(x, results.best_fit, label="Fit")
    ax.legend()

    # Show figure in Streamlit
    st.pyplot(fig)
