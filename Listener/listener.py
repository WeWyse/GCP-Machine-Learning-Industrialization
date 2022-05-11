from tweepy import StreamingClient, StreamRule
from google.cloud import pubsub_v1
import tweepy
import json

# Pub/Sub topic configuration
publisher 	= pubsub_v1.PublisherClient()
topic_path 	= publisher.topic_path("rare-result-248415","Tweets_topic_test")

# Authenticate to the API
bearer_token = "AAAAAAAAAAAAAAAAAAAAAA2TcQEAAAAAYqjaUkY1vN4803JuUXedqMSypSc%3Dm6pjmutLIY2XRRqmtMCMMnPbskVQkjHDM2TBPE4ff9WlAtQQgt"



# Define the list of terms to listen to
lst_hashtags = ["#macron"]

# Method to push messages to pub/sub
def write_to_pubsub(data):
    try:
        if data["lang"] == "en":
            publisher.publish(topic_path, data=json.dumps({
                "text"	    : 	data["text"],
                "user_id"   : 	data["user_id"],
                "id"        : 	data["id"],
                "created_at": 	data["created_at"]
            }).encode("utf-8"), tweet_id=str(data["id"]).encode("utf-8"))
    except Exception as e:
        raise

# Method to format a tweet from tweepy
def reformat_tweet(tweet):
    x = tweet

    processed_doc = {
        "id"					: x["id"],
        "lang"					: x["lang"],
        "retweeted_id"			: x["retweeted_status"]["id"] if "retweeted_status" in x else None,
        "favorite_count" 		: x["favorite_count"] if "favorite_count" in x else 0,
        "retweet_count"			: x["retweet_count"] if "retweet_count" in x else 0,
        "user_id"				: x['data']["author_id"],
        "created_at"			: x["created_at"].strftime( "%m/%d/%Y, %H:%M:%S")
    }


    if "extended_tweet" in x:
        processed_doc["text"] 			= x["extended_tweet"]["full_text"]
    elif "full_text" in x:
        processed_doc["text"] 			= x["full_text"]
    else:
        processed_doc["text"]			= x["text"]

    return processed_doc



# Custom listener class
class TweetPrinterV2(tweepy.StreamingClient):

    def on_tweet(self, tweet):
        write_to_pubsub(reformat_tweet(tweet))
        print(reformat_tweet(tweet))
        print("-"*50)

# Start listening

printer = TweetPrinterV2(bearer_token)

# add new rules
rule = StreamRule(value="#macron")
printer.add_rules(rule)

printer.filter(expansions=["author_id"], tweet_fields=["created_at","entities","lang"])