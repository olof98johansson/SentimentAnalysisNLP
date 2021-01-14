<h1 align="center"> Project Overview </h1>
This project aims to build different NLP models for sentiment analysis and classify depression in tweets with the goal of forecasting the frequency of these from before and after the Covid-19 outbreak. This will both test the knowledge of different NLP methods and analyze the performance of these using naive bench-marking as well as implementing these in the new problem of sentiment analysis. Furthermore, this project will also contribute to the research of the social implications and impact on mental health due to the Covid-19 pandemic.
<hr>
<h2 align="center"> To-do Checklist </h2>

| 📌 Checkpoint                                              | Status |
| ------------------------------------------------- | ----   |
| ◾ <input type="checkbox" disabled checked /> Define Keywords  |  :heavy_check_mark:  |
| ◾ <input type="checkbox" disabled  checked/>  Tweet scraping |  :heavy_check_mark:    |
| ◾ <input type="checkbox" disabled  checked/> Text processing |  :heavy_check_mark:    |
| ◾ <input type="checkbox" disabled  checked/>  Preprocessing |  :heavy_check_mark:   |
| ◾ <input type="checkbox" disabled  checked/> Implement NLP models |  :clock930: :heavy_check_mark:   |
| ◾ <input type="checkbox" disabled  checked/> Benchmarking |   :no_entry_sign:   |
| ◾ <input type="checkbox" disabled  checked/> Visualizations |  :clock930:    |
| ◾ <input type="checkbox" disabled  checked/> Test predictions |  :heavy_check_mark:    |
| ◾ <input type="checkbox" disabled  checked/> Forecasting | :heavy_check_mark:    |

<hr>

<h2 align="center"> Results of training session with LSTM model </h2>

![First training session](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/training_animation_progress_REAL.gif?raw=true)
<br>

<h1 align="center"> Results </h2>
The results show the forecast of the percentage of depressive tweets, weekly collected, predicted by the LSTM model.

<h2 align="center"> From three months before UK initial lockdown to three months after </h2>

![First forecast](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_orig.png?raw=true)
<br>

<h2 align="center"> Same time period for the previous year </h2>

![Second forecast](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_year_before.png?raw=true)
<br>

<h2 align="center"> From three months after the initial UK lockdown to recent (dec 2020) </h2>

![Third forecast](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_up_to_now.png?raw=true)

<br>

<h2 align="center"> Comparison of them all </h2>

![Comparison](https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/comparison.png?raw=true)

<hr>


<h2 align="center"> Animated time series of the forecasts </h2>

<table>
  <tr>
    <td>3 months before and after UK lockdownn</td>
     <td>Same period previous year</td>
     <td>3 months after lockdown to recent</td>
  </tr>
  <tr>
    <td><img src="https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_bar_race_orig.gif" width=270></td>
    <td><img src="https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_bar_race_last_year.gif" width=270></td>
    <td><img src="https://github.com/olof98johansson/SentimentAnalysisNLP/blob/main/plots/forecast_bar_race_up_to_now.gif" width=270></td>
  </tr>
 </table>


<h2 align="center"> Example of using twint </h2>
Running the test function in the twint script with json output format, the data is saved as belowed:
<br>
<br>
{"id": 1338180975268925443, "conversation_id": "1338180975268925443", "created_at": "2020-12-13 18:56:21 Västeuropa, normaltid", "date": "2020-12-13", "time": "18:56:21", "timezone": "+0100", "user_id": 1300253484915269640, "username": "dayzdevious", "name": "❄️rejoice 📌rt pinned❄️", "place": "", "tweet": "If someone tells you that they’re so depressed that they want to die and you tell them it’s bc of their phone??? You’re going to burn in hell!!", "language": "en", "mentions": [], "urls": [], "photos": [], "replies_count": 1, "retweets_count": 0, "likes_count": 0, "hashtags": [], "cashtags": [], "link": "https://twitter.com/DayzDevious/status/1338180975268925443", "retweet": false, "quote_url": "", "video": 0, "thumbnail": "", "near": "", "geo": "", "source": "", "user_rt_id": "", "user_rt": "", "retweet_id": "", "reply_to": [], "retweet_date": "", "translate": "", "trans_src": "", "trans_dest": ""}
<br>
{"id": 1338180974996316167, "conversation_id": "1338160462064721922", "created_at": "2020-12-13 18:56:21 Västeuropa, normaltid", "date": "2020-12-13", "time": "18:56:21", "timezone": "+0100", "user_id": 1289143207482331138, "username": "asim_crash", "name": "Asim | Banooca Boi", "place": "", "tweet": "@koifishbesties Damn it autocorrect! I meant why would YOU be depressed? I don’t want you to be sad :(", "language": "en", "mentions": [], "urls": [], "photos": [], "replies_count": 0, "retweets_count": 0, "likes_count": 0, "hashtags": [], "cashtags": [], "link": "https://twitter.com/Asim_Crash/status/1338180974996316167", "retweet": false, "quote_url": "", "video": 0, "thumbnail": "", "near": "", "geo": "", "source": "", "user_rt_id": "", "user_rt": "", "retweet_id": "", "reply_to": [{"screen_name": "koifishbesties", "name": "KoiFish||", "id": "1268632343297826816"}], "retweet_date": "", "translate": "", "trans_src": "", "trans_dest": ""}
<br>
◾
<br>
◾
<br>
◾
<br>
◾
<br>
◾
