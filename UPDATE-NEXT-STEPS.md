# NLP Tweet Analysis
## Useful Resources

### Data Collection and Methods:
 *Based on the resources below, we could finalize the data collection and pre-processing steps for our project. Once done, I believe that we can try to feed the models from our assignments directly and see how they behave with our data*

1. https://github.com/swcwang/depression-detection : The readme in this resource gives a nice view of the kind of tweets to consider and how to narrow down and clean the tweets of mentions, hashtags etc.
	 - We also need to define the time range of tweets we will consider. 
	 - If we are not using specific words from tweets, then it is important to limit the scope of search if too many tweets start getting loaded.
2. Resources at https://github.com/peijoy/DetectDepressionInTwitterPosts and https://github.com/weronikazak/depression-tweets suggests creating a  dataset which consists of tweets specific to depression scraped from the internet, and then scrape random tweets and form a combined dataset to train a model to see how well it can detect depression. 
3. https://www.academia.edu/40297773/Evaluating_Mental_Health_using_Twitter_Data also gives a detailed approach of the above, highlighting the data collection and processing.
4. https://doi.org/10.1007/s11606-020-05988-8 : This paper " Tracking Mental Health and Symptom Mentions on Twitter During COVID-19" analyses (a) Sentiment, (b) stress, (c) anxiety, and (d) loneliness expressions derived from data-driven machine learning models on Twitter language from the start of January till May 6 in 2019  and 2020. We could do something similar, maybe with UK (if possible) instead of USA (European country with majority English tweets) .
5. https://www.ssrn.com/abstract=3383359 : A very different idea is highlighted in this paper, where it samples a set of users and analyses their tweets over a period of time and monitoring their tweeting pattern, behavior, Language, sentiment etc., to detect of the user may be diagnosed by anxious depression. This is probably not in the direction we are looking at our problem, but an interesting read nevertheless. 


**OTHER THOUGHTS and Potential Next Steps:** 
 - We should source the data as csv files.
 - It would also be useful to see specifically which subset of twitter attributes we would require, so that maybe the dataset is smaller in size and could let us maybe  download a larger list of tweets, with lesser columns.
 - Finalize 1 to 2 NLP Models we will use to perform the analysis. 

