# Fake2vec

![MacDown logo](http://macdown.uranusjr.com/static/images/logo-160.png)

Stopping the spread of fake news at the source

Based on available resources from the [Media Bias/Fact Check](http://www.mediabiasfactcheck.com) website.


Options
-----------------------------------------------------------------------------------------------
The main scripts listed below each have options (defaults are set). To see the options type
python one_of_the_scripts_below.py --help


Train and deploy fake2vec
-----------------------------------------------------------------------------------------------
Scrape the publisher list provided by MSFC:
python utils/scrape_publisher.py

Prepare the scraped data for training:
python utils/text_processor.py

Train the doc2vec embedding:
python utils/train_doc2vec.py

Train the neural classifier model:
python utils/train_classifier.py

Deploy fake2vec on your local server:
cd app; python main.py
=============================================
