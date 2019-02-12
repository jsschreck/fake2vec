from flask import Flask, Response, render_template, request, redirect, url_for

from deepspot.utils.text_processor import normalize
from deepspot.utils.scraper import load_url
from deepspot.utils.model import main

import io, random, base64, urllib, sys, re

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import pylab

import tensorflow as tf

app = Flask(__name__)

####################  Home-page   ##########################
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':  #this block is only entered when the form is submitted

        query_text = request.form.get('text-query')
        query_url = request.form.get('url-query')

        if query_text and not query_url:
            # Clean the text
            query = query_text
        elif not query_text and query_url:
            query = load_url(query_url)
        else:
            print("You need to enter text, or a URL. Going home.")
            # Redirect to "failed" page. From there redirect back to search again
            return redirect(url_for('home'))

        with open("static/query.txt", "w") as fid:
            fid.write("{}".format(query))
        return redirect(url_for('result'))

    return render_template("index.html")

####################  Isolated search page   ##########################
@app.route('/text/', methods=['GET', 'POST'])
def article_query():
    if request.method == 'POST':  #this block is only entered when the form is submitted

        query_text = request.form.get('text-query')
        query_url = request.form.get('url-query')

        if query_text and not query_url:
            # Clean the text
            query = query_text
        elif not query_text and query_url:
            query = load_url(query_url)
        else:
            print("You need to enter text, or a URL. Going home.")
            # Redirect to "failed" page. From there redirect back to search again
            return redirect(url_for('home'))

        with open("static/query.txt", "w") as fid:
            fid.write("{}".format(query))
        return redirect(url_for('result'))

    return render_template("text.html")

####################  About doc2vec   ##########################
@app.route('/about/', methods=['GET', 'POST'])
def about_fake2vec():
    return render_template("about.html")
####################  Call  doc2vec   ##########################
@app.route('/query/', methods=['GET', 'POST'])
def try_another_query():
    return render_template("text.html")

##############################################
def create_fig(labels, sizes, tag):
    params = {
          'backend': 'ps',
		  'lines.markersize'  : 6,
	      'axes.labelsize': 12,
	      'legend.fontsize': 12,
	      'xtick.labelsize': 12,
	      'ytick.labelsize': 12,
	      'font.serif'    : 'Roboto',
	  	  'font.sans-serif': 'Roboto',
	      'text.usetex': True,
	      'figure.dpi'       : 600}
    # Figure
    fig = Figure()
    fig.set_tight_layout(True)
    canvas = FigureCanvas(fig)

    pylab.rcParams.update(params)
    ax = fig.add_subplot(111)

    # Custom color spectrum
    if len(sizes) == 3:
        values = ['HIGH', 'MIXED', 'LOW']
        colors = ["g", 'y', 'r']
    else:
        values = ['extreme-left', 'left', 'left-center', 'center', 'right-center', 'right', 'extreme-right']
        #cmap = matplotlib.cm.get_cmap('seismic')
        #z = list(range(len(sizes)))
        #normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
        #colors = [cmap(normalize(value)) for value in z]
        colors = [  (0.0, 0.0, 0.3, 1.0),
                    (0.0, 0.0, 0.7611764705882352, 1.0),
                    (0.3333333333333333, 0.3333333333333333, 1.0, 1.0),
                    '#dddddd', #(1.0, 0.9921568627450981, 0.9921568627450981, 1.0),
                    (1.0, 0.33333333333333337, 0.33333333333333337, 1.0),
                    (0.8294117647058823, 0.0, 0.0, 1.0),
                    (0.5, 0.0, 0.0, 1.0)    ]

    colors = [colors[values.index(x)] for x in labels]

    def my_autopct(pct):
        return ('%1.1f%%' % pct) if pct > 0.5 else ''

    def my_list(data, labs):
        list = []
        for i in range(len(data)):
            if (data[i]*100.0/sum(data)) > 2: #2%
                list.append(labs[i])
            else:
                list.append('')
        return list

    # Pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=my_list(sizes, labels),
                                        colors=colors,
                                        autopct=my_autopct,
                                        shadow=False,
                                        startangle=140,
                                        wedgeprops={'alpha':0.5})
    canvas = FigureCanvas(fig)
    #canvas.figure.tight_layout()
    png_output = io.BytesIO()
    #plt.savefig(png_output, format='png')
    #png_output.seek(0)
    canvas.print_png(png_output, dpi=600)
    data = base64.b64encode(png_output.getvalue()).decode('utf-8')
    data_url = 'data:image/png;base64,{}'.format(urllib.parse.quote(data))
    #png_output.close()
    #plt.close(fig)
    return data_url

####################  Call results page   ##########################
@app.route("/result/", methods=['GET', 'POST'])
def result():
    # Read text file
    with open("static/query.txt", "r") as fid:
        query = " ".join(fid.readlines())
    query = normalize(query.split(" "))
    cleaned_query = " ".join(query)

    # Get fact2vec results from query
    pubs_decode, facts, affiliation = main(query)

    # Summary field
    if 'LOW' in facts[0][0] and 'extreme' in affiliation[0][0]:
        summary = "WARNING! Article could be fake news."
    elif 'MIXED' in facts[0][0]:
        if 'left' == affiliation[0][0] or 'right' == affiliation[0][0]:
            summary = "Article may have factual inconsistencies and a political lean."
        else:
            summary = "Article may have factual inconsistencies but does not appear to have a strong political lean."
    else:
        summary = "Article appears factually correct and does not have a strong political lean."

    # facts figure
    labels, sizes = zip(*facts)
    facts_fig = create_fig(labels, sizes, "img1")

    # affil figure
    labels, sizes = zip(*affiliation)
    affil_fig = create_fig(labels, sizes, "img2")

    # load into result page
    return render_template("result.html",
                            query = cleaned_query,
                            summary = summary,
                            result1 = facts_fig,
                            result2 = affil_fig,
                            publisher = pubs_decode,
                            fact_input = facts,
                            bias_input = affiliation)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
