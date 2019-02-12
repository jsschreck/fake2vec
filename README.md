# Fake2vec

<!-- ![MacDown logo](http://macdown.uranusjr.com/static/images/logo-160.png) -->

Stopping the spread of fake news at the source

Based on available publisher analysis from the [Media Bias/Fact Check](http://www.mediabiasfactcheck.com) website.


Options
-----------------------------------------------------------------------------------------------
The main scripts listed below each have options (with defaults). To see the options type

```
python one_of_the_scripts_below.py --help
```

Train and deploy fake2vec
-----------------------------------------------------------------------------------------------
Launch the following script to create some directories to save data/models. It will then call the relevant scripts to train the doc2vec word embedding and the classifier.

```
sh launch.sh
```
-----------------------------------------------------------------------------------------------
You can also do it one step at a time. First, make some directories:

```
mkdir data models reports results
mkdir models/classifier models/doc2vec
```

Next, scrape the publisher list provided by MSFC:

```
python utils/scrape_publisher.py
```

Prepare the scraped data for training:

```
python utils/text_processor.py
```

Train the doc2vec embedding:

```
python utils/train_doc2vec.py
```

Train the neural classifier model:

```
python utils/train_classifier.py
```

Deploy fake2vec on your local server:

```
cd app
python main.py
```
=============================================
