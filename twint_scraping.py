# NOTE: TWINT NEEDS TO BE INSTALLEED BY THE FOLLOWING COMMAND:
# pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
# OTHERWISE IT WON'T WORK


import twint
import nest_asyncio
nest_asyncio.apply()
from dateutil import rrule
from datetime import datetime, timedelta

def get_weeks(start_date, end_date):
    '''
    Finds collection of weeks chronologically from a starting date to a final date

    Input: start_date - date of which to start collecting with format [year, month, day] (type: list of ints)
           end_date - date of which to stop collecting with format [year, month, day] (type: list of ints)

    Output: weeks - list containing the lists of starting and ending date for each week with format
                    "%Y-%m-%d %h-%m-%s" (type: list of lists of strings)
    '''
    start_year, start_month, start_day = start_date
    final_year, final_month, final_day = end_date
    start = datetime(start_year, start_month, start_day)
    end = datetime(final_year, final_month, final_day)
    dates = rrule.rrule(rrule.WEEKLY, dtstart=start, until=end)
    nr_weeks = 0
    for _ in dates:
        nr_weeks+=1
    weeks = []
    for idx, dt in enumerate(dates):
        if idx < nr_weeks-1:
            week = [dates[idx].date().strftime('%Y-%m-%d %H:%M:%S'),
                    dates[idx+1].date().strftime('%Y-%m-%d %H:%M:%S')]
            weeks.append(week)

    return weeks



def collect_tweets(keywords = None, nr_tweets = None,
                   output_file=None, coord=None, timespan=[None, None]):
    '''
    Collectiing tweets using twint based on different attributes and save to json file

    Input: keywords - keywords that the tweet should contain (type: string)
           nr_tweets - number of tweets to collect (type: int)
           output_file - path and name to where the file should be saved (type: string, extension: .json)
           near - location or city of which the tweets were tweeted (type: string)
           timespan - timespan of when the tweet was tweeted in format "%Y-%m-%d %h-%m-%s" (type: string)

    Output: Returns twint object
    '''
    # configuration
    config = twint.Config()
    # Search keyword
    config.Search = keywords
    # Language
    config.Lang = "en"
    # Number of tweets
    config.Limit = nr_tweets
    #Dates
    config.Since = timespan[0]
    config.Until = timespan[1]
    # Output file format (alternatives: json, csv, SQLite)
    config.Store_json = True
    # Name of output file with format extension (i.e NAME.json, NAME.csv etc)
    config.Output = output_file

    config.Geo = coord

    # running search
    twint.run.Search(config)
    return twint


# EXAMPLE
def test():
    config = twint.Config()
    config.Search = None
    config.Near = "london"
    config.Lang = "en"
    config.Limit = 10
    config.Since = "2016-10-29 00:00:00"
    config.Until = "2016-11-29 12:15:19"
    config.Store_json = True
    config.Output = "test2.json"

    #running search
    twint.run.Search(config)


#test()
