<h1 align="center"> Description </h1>
To run the files, make sure all the dependencies are installed. Do this by running

```
pip install -r requirements.txt
```

in the terminal. Then to run the training process for the LSTM model along with the data collection, data cleaning and preprocessing, type

```
python -m main.py
```

and then to collect the test data as well as perform the weekly based predictions with visualizations, type

```
python -m predict.py
```

#### Below is a short, partial discription about the project itself including methodology and results 

<hr>

<h2 align="center"> Sentiment Analysis to Classify Depression from Twitter Using Tweets Before andAfter COVID-19 Through Different NLP Approaches </h2>

<b>[Camille Porter](https://github.com/finli), [Olof Johansson](https://github.com/olof98johansson), [Savya Sachi Gupta](https://github.com/foo-bar-omastar)</b>

Independent project in course <b>DAT450: Machine Learning for Natural Language Processing</b>

Chalmers Institute of Technology, Sweden


<h3 align="center"> Abstract </h3>
We aim to understand the affect of the COVID-19 pandemic on mental health, specifically depression, by performing sentiment analysis on ‘tweets’ shared on the social media service  Twitter.  We  selected  the  United  Kingdom  and  Ireland for our analysis as they had a government instituted lockdown across the country, which provides us with a definitive date as a reference point to gauge trends before and after. In order to understand how a lockdown affects depression, we sampled tweets from these locations and trained two different models to detect depression in tweets — a LSTM model, and theDistilBERT  model, which is a condensed form of BERT. We scraped 5,000 tweets for each week that we measured during multiple time periods. The LSTM model performed better than DistilBERT yielding an accuracy of 0.94 as compared to 0.90 for DistilBERT. We found a 2-3% bump in the levelof depression two weeks after lockdown started, but no longterm changes.

<br>




# Methodology <a name="Methodology"></a>
## Data Collection <a name="Data Collection"></a>
In order to maximize the locations included in our analysis, we used the longitude/latitude method. Using Google Earth, we determined that the Isle of Man is approximately at the center of the UK. Using the measurement tool on Google Earth, we found that a 550 kilometer circle around the Isle of Man covered all of UK and Ireland without touching France. In order to train the model we sourced tweets in two halves. One half of tweets related to depression, and another half of tweets that were non-depressive. This enables accurate labeling of data that helps train the model. The words we used to label depressive tweets were : depressed, lonely, sad, depression, tired, and anxious. The words that were labeled non-depressive were: happy, joy, thankful, health, hopeful, and glad. For each word specified above, we scraped 1,000 tweets, resulting in a training set size of 12,000 tweets. 80% of the tweets were selected for training and 20% for testing. Subsequently, for analyzing, we sourced 5,000 tweets per week for three different time periods; three months before and after the initial UK lockdown of 23 March, 2020, same period the year before and then a six months period starting from three months after the initial lockdown up to 17th of December, 2020. The code for the tweet collection process is found in the [twint_scraping.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/twint_scraping.py) file.

## Preprocessing <a name="Preprocessing"></a>
Prior to the training process, the collected tweets had to be cleaned, annotated, and combined as many raw collected tweets contain emojis, URLs, mentions and non-alphabetic
tokens. Thus, a pre-processing step cleaned the tweets as above and also removed possible punctuation marks. Next, a vocabulary based on words from the training data was built which consisted of two dictionaries for encoding and decoding the input text data. Moreover, the encoding process also padded the input sequences to be the same length by adding a specific padding token. Additionally had the labels also be encoded as the training data were labeled with either depressive or not-depressive. These two categories were encoded into corresponding integers of 0 and 1. This code for these processes are found in the [data_cleaning.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/data_cleaning.py) file and the [preprocessing.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/preprocessing.py)

## Models <a name="models"></a>
The initial model used for the sentiment analysis was a standardized embedding, that converts input integer tokens into real-valued data, followed by a Long-Short-Term-Memory (LSTM) model with an added feed-forward layer with dropout as output layer. The initial states, $h0$ and $c0$, of the LSTM network were zero initiated and the output of the feed-forward output layer was then fed into a sigmoid function, also used as the activation function in the LSTM network. As the problem is binary classification, the loss was thereafter computed by the binary cross entropy loss (BCE). Lastly, the decoding of the output from the predictions on the test data was done as

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{decode}(\hat{y})&space;=&space;\begin{cases}&space;\text{\textit{depressive}},&space;\quad&space;\text{if&space;$\hat{y}_i&space;<&space;0.5$}\\&space;\text{\textit{not-depressive}},&space;\quad&space;\text{if&space;$\hat{y}_i&space;\geq&space;0.5$}\end{cases}" target="_blank"><img align="center" src="https://latex.codecogs.com/gif.latex?f_{decode}(\hat{y})&space;=&space;\begin{cases}&space;\text{\textit{depressive}},&space;\quad&space;\text{if&space;$\hat{y}_i&space;<&space;0.5$}\\&space;\text{\textit{not-depressive}},&space;\quad&space;\text{if&space;$\hat{y}_i&space;\geq&space;0.5$}\end{cases}" title="f_{decode}(\hat{y}) = \begin{cases} \text{\textit{depressive}}, \quad \text{if $\hat{y}_i < 0.5$}\\ \text{\textit{not-depressive}}, \quad \text{if $\hat{y}_i \geq 0.5$}\end{cases}" /></a>


The code for the LSTM model and the training function is found in [models.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/models.py) file and [train.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/train.py) file respectively.
<br>

In addition to our LSTM model, we wanted to try a stateof-the-art transfer learning model. We decided to use DistilBERT, a smaller version of Bidirectional Encoder Representation from Transformers (BERT). BERT is a bidirectional LSTM with multi-head attention. The model implementation and training for the DistilBERT is found in the [Twitter_Classification_DistilBERT.ipynb](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/Twitter_Classification_DistilBERT.ipynb) file.


# Results <a name="Results"></a>
## Training <a name="Training"></ha>
The code to run the training process of the LSTM model is found in [main.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/main.py) file. The results of the training progress and accuracy metrics for the LSTM model are shown below. The highest validation accuracy for this model was 0.94.

![First training session](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/training_animation_progress_REAL.gif?raw=true)
<br>

The corresponding results for the DistilBERT model is further shown [here](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/distilBERT_plots/8_epoch_train.pdf).


Since our LSTM model has a greater validation accuracy, we will select that model performing our time period analysis in the next section.

## Time period analysis <a name="forecast"></a>
The results show the forecast of the percentage of depressive tweets, weekly collected, predicted by the LSTM model. From three months before UK initial lockdown to three months after, the following results were obtained. The code for making the predictions is found in the [predict.py](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/predict.py) file.

![First forecast](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_orig.png?raw=true)
<br>

The results of the same time period for the previous year is further shown below.

![Second forecast](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_year_before.png?raw=true)
<br>

The final analysis from three months after the start of the initial UK lockdown up to the 17th of December, 2020, is shown below.
![Third forecast](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_up_to_now.png?raw=true)

<br>

The combined result, for comparison, is further shown below.

![Comparison](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/comparison.png?raw=true)

An animation of the weekly results are also visualized below.


<h2 align="center"> Animated time series of the forecasts </h2>

<table>
  <tr>
    <td>3 months before and after UK lockdownn</td>
     <td>Same period previous year</td>
     <td>3 months after lockdown to recent</td>
  </tr>
  <tr>
    <td><img src="https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_bar_race_orig.gif" width=300></td>
    <td><img src="https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_bar_race_last_year.gif" width=300></td>
    <td><img src="https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_bar_race_up_to_now.gif" width=300></td>
  </tr>
 </table>
