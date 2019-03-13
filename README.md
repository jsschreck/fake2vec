# Fake2vec

<img src="app/static/logo_transparent.png" alt="drawing" width="200"/>

Stopping the spread of fake news at the source

Based on available publisher analysis from the [Media Bias/Fact Check](http://www.mediabiasfactcheck.com) website.


Options
-----------------------------------------------------------------------------------------------
The main scripts listed below each have options (with defaults). To see the options type

```
python [one_of_the_scripts_below.py] --help
```

Train and deploy fake2vec
-----------------------------------------------------------------------------------------------
Launch the following shell script to train a doc2vec word embedding and a neural classifier.

```
sh launch.sh
```
-----------------------------------------------------------------------------------------------
You can also do it one step at a time. First, scrape the publisher list provided by MSFC:

```
python utils/scrape_publisher.py
```

Next, prepare the scraped data for training:

```
python utils/text_processor.py
```

Now train the doc2vec embedding:

```
python utils/train_doc2vec.py
```

Then train a classifier model:

```
python utils/train_classifier.py
```

Deploy fake2vec on your local server:

```
cd app
python main.py
```
=============================================
