import numpy as np
from scipy.stats import gamma
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.layouts import column

def plot_gamma_distribution(k, theta, x_range=(0, 20), num_points=1000, num_samples=1000):
    """
    Plot the PDF of a gamma distribution with shape parameter k and scale parameter theta.
    Also plot a histogram of the integer parts of samples drawn from the gamma distribution.

    :param k: Shape parameter for the gamma distribution
    :param theta: Scale parameter for the gamma distribution
    :param x_range: Range of x values for the plot
    :param num_points: Number of points to plot
    :param num_samples: Number of samples to draw for the histogram
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = gamma.pdf(x, k, scale=theta)

    p1 = figure(title=f"Gamma Distribution (k={k}, θ={theta})", x_axis_label='x', y_axis_label='PDF')
    p1.line(x, y, legend_label=f"Gamma PDF (k={k}, θ={theta})", line_width=2)
    p1.legend.location = "top_right"

    # Draw samples from the gamma distribution
    samples = gamma.rvs(k, scale=theta, size=num_samples)
    integer_samples = np.floor(samples).astype(int)

    # Create a histogram of the integer parts of the samples
    hist, edges = np.histogram(integer_samples, bins=np.arange(x_range[0], x_range[1] + 1))

    p2 = figure(title="Histogram of Integer Parts of Samples", x_axis_label='Value', y_axis_label='Frequency')
    p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.7)

    layout = column(p1, p2)

    output_file("gamma_distribution.html")
    show(layout)

if __name__ == "__main__":
    k = 2.0  # Example shape parameter
    theta = 3  # Example scale parameter
    plot_gamma_distribution(k, theta)
