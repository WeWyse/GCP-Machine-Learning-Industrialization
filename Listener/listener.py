from tweepy import StreamingClient, StreamRule
from google.cloud import pubsub_v1
import tweepy
import json
import yaml

with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
    PROJECT_ID = cfg['project_id']
    TOPIC_NAME = cfg['topic_name']
    BEARER_TOKEN = cfg['bearer_token']
    HASHTAG = cfg['hashtag']

# Pub/Sub topic configuration
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)
# Authenticate to the Twitter API
bearer_token = BEARER_TOKEN
# Define the list of terms to listen to on Twitter
lst_hashtags = HASHTAG

# Method to push messages to pub/sub
def write_to_pubsub(data):
    try:
        if data["lang"] == "en":
            publisher.publish(topic_path, data=json.dumps({
                "text": data["text"],
                "user_id": data["user_id"],
                "id": data["id"],
                "created_at": data["created_at"]
            }).encode("utf-8"), tweet_id=str(data["id"]).encode("utf-8"))
    except Exception as e:
        raise

# Method to format a tweet from tweepy
def reformat_tweet(tweet):
    processed_doc = {
        "id": tweet["id"],
        "lang": tweet["lang"],
        "retweeted": tweet["retweeted_status"]["id"] if "retweeted_status" in tweet else None,
        "favorite_co": tweet["favorite_count"] if "favorite_count" in tweet else 0,
        "retweet_co": tweet["retweet_count"] if "retweet_count" in tweet else 0,
        "user_id": tweet['data']["author_id"],
        "created": tweet["created_at"].strftme( "%m/%d/%Y, %H:%M:%S")
    }
    if "extended_tweet" in tweet:
        processed_doc["text"] = tweet["extended_tweet"]["full_text"]
    elif "full_text" in tweet:
        processed_doc["text"] = tweet["full_text"]
    else:
        processed_doc["text"] = tweet["text"]
    return processed_doc

# Custom listener class
class TweetPrinterV2(tweepy.StreamingClient):

    def on_tweet(self, tweet):
        write_to_pubsub(reformat_tweet(tweet))
        print(reformat_tweet(tweet))
        print ("- " *50)

# Start listening
printer = TweetPrinterV2(bearer_token)

# add new rules
rule = StreamRule(value="#macron")
printer.add_rules(rule)
printer.filter(expansions=["author_id"], tweet_fields=["creae d_at","eni ties","lang"])
