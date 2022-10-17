from matplotlib.dates import DateFormatter
DATE_FORMATTER = DateFormatter("%b. %d")

CROP_TYPE_COLOR_MAPPING = {"grassland":"g", "corn": "y", "summer barley": "r", "winter barley": "k", "winter wheat": "c", "fallow":"m"}
COLOR_MAPPING_PLOTLY = {"grassland":"green", "corn": "yellow", "summer barley": "red", "winter barley": "black", "winter wheat": "dodgerblue", "fallow":"brown"}

FIGURES_BASE_PATH = "C:/Users/Ivica Obadic/paper_plots/crop-type-classification-explainability/"
tex_fonts = {
    #source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    #"axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    'text.latex.preamble': r"\usepackage{amsmath}"
}

poster_fonts = {
    #source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 20,
    "font.size": 20,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.titlesize" : 20,
}

attn_weight_matrix_viz_fonts = {
    #source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize" : 10,
    'text.latex.preamble': r"\usepackage{amsmath}"
}