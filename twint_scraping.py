# NOTE: TWINT NEEDS TO BE INSTALLEED BY THE FOLLOWING COMMAND:
# pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
# OTHERWISE IT WON'T WORK


import twint
import nest_asyncio
nest_asyncio.apply()

def collect_tweets(keywords = None, nr_tweets = None, output_file=None):
    # configuration
    config = twint.Config()
    # Search keyword
    config.Search = keywords
    # Language
    config.Lang = "en"
    # Number of tweets
    config.Limit = nr_tweets
    #Dates
    # config.Since = "2019–11–29 11:01:01"
    # config.To = "2020–11–29 11:01:01"
    # Output file format (alternatives: json, csv, SQLite)
    config.Store_json = True
    # Name of output file with format extension (i.e NAME.json, NAME.csv etc)
    config.Output = output_file

    # running search
    twint.run.Search(config)
    return twint


# EXAMPLE
def test():
    config = twint.Config()
    config.Search = "depressed"
    config.Lang = "en"
    config.Limit = 10
    #config.Since = "2019–11–29 11:01:01"
    #config.To = "2020–11–29 11:01:01"
    config.Store_json = True
    config.Output = "test.json"

    #running search
    twint.run.Search(config)


test()