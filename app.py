import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import zipfile
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import  matplotlib.pyplot as plt
import plotly.express as px

st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets about US Airlines")

st.markdown("This app is a Streamlit dashboard to analyze the sentiment of Tweets 🐦")
st.sidebar.markdown("This app is a Streamlit dashboard to analyze the sentiment of Tweets 🐦")

# Google Drive URL for the dataset (direct download link)
google_drive_url = "https://drive.google.com/uc?id=1-lND-JSWbIxmBuUiQW8xCVerKORipG9H&export=download"

@st.cache(persist=True)
def load_data():
    response = requests.get(google_drive_url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    # Assuming your dataset is a CSV file inside the zip
    csv_file = zip_file.extract(zip_file.namelist()[0], path='./data')  # Extract to a local directory
    data = pd.read_csv(csv_file, error_bad_lines=False, warn_bad_lines=True, encoding='utf-8')
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

st.sidebar.subheader("Show a random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
#random_tweet = data['airline_sentiment'].sample(n=1).iloc[0]
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])



# Plotting bar and pie chart
st.sidebar.subheader("Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualisation Type', ['Histogram', 'Pie Chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame(
    {'Sentiment': sentiment_count.index, 'Tweets': sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown('###Number of tweets by sentiments')
    if select == "Histogram":
        fig = px.bar(sentiment_count, x = 'Sentiment', y = 'Tweets', color = 'Tweets', height = 500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values = 'Tweets', names = 'Sentiment')
        st.plotly_chart(fig)




# Plotting no of tweets by sentiment for each airline
st.sidebar.subheader("Total number of tweets by sentiment for each airline")
airline = st.sidebar.multiselect(
    "Pick Airlines",('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'),key = '0')

if len(airline) > 0:
    data = data[data.airline.isin(airline)]
    fig_2 = px.histogram(data, x = 'airline', y = 'airline_sentiment', histfunc='count',
                         color = 'airline_sentiment', facet_col='airline_sentiment', labels = {'airlien_sentiment':'tweets'}, height=600, width = 800)
    st.plotly_chart(fig_2)



# Word Cloud
st.sidebar.subheader("Word Cloud")
word_sentiment = st.sidebar.radio(
    'Display wordcloud for what sentiment?', ('positive','neutral','negative')
)
if not st.sidebar.checkbox("Close", True, key = '3'):
    st.subheader("Word cloud for %s sentiment"%(word_sentiment))
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ''.join([word for word in words.split()
                               if 'http' not in word and not word.startswith('@') and word!= 'RT'])
    # Creating a wordcloud object
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width = 800, height = 640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


#Extra features
    # Sentiment Over Time
st.subheader("Sentiment Over Time")
line_chart_data = data.groupby([data['tweet_created'].dt.date, 'airline_sentiment']).size().unstack()
st.line_chart(line_chart_data)

# Word Frequency Analysis
st.subheader("Word Frequency Analysis")
word_freq_sentiment = st.radio("Display word frequency for what sentiment?", ('positive', 'neutral', 'negative'))
if not st.checkbox("Hide", True, key='word_freq_checkbox'):
    word_freq_df = data[data['airline_sentiment'] == word_freq_sentiment]
    word_freq = pd.Series(' '.join(word_freq_df['text']).split()).value_counts()[:20]
    st.write(word_freq)

# Custom Word Cloud
st.subheader("Custom Word Cloud")
custom_word = st.text_input("Enter a word for custom word cloud")
if st.button("Generate Word Cloud") and custom_word:
    custom_word_df = data[data['text'].str.contains(custom_word, case=False)]
    words = ' '.join(custom_word_df['text'])
    processed_words = ''.join([word for word in words.split()
                               if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    st.image(wordcloud.to_array())

# Sentiment Analysis of Custom Text
st.subheader("Sentiment Analysis of Custom Text")
custom_text = st.text_area("Enter text for sentiment analysis")
if st.button("Analyze Sentiment") and custom_text:
    # Perform sentiment analysis on custom text
    # sentiment = perform_sentiment_analysis(custom_text)
    # st.write("The sentiment of the custom text is:", sentiment)
    st.write("Sentiment analysis feature is under construction.")

# Topic Modeling
st.subheader("Topic Modeling")
st.write("Topic modeling feature is under construction.")

# Interactive Filters
st.subheader("Interactive Filters")
st.write("Interactive filters feature is under construction.")
