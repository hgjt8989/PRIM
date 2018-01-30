#!/usr/bin/python
import traceback
import time
import os
import json
import sys
import codecs
import datetime
import subprocess
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

class StdOutListener(StreamListener):
    def __init__(self, path, track_list):
        self.today = datetime.date.today()
        self.path = path
        self.track_list = track_list
        self.file = codecs.open(self.path+str(self.today), 'a', "utf-8")
        self.flag = False
    def on_data(self, data):   
        json_tweet = json.loads(data)
        jkeys = list(json_tweet.keys())
        if 'text' in jkeys:
            text = json_tweet['text']
            for i in self.track_list:
                if i in text:  
                    tweet_data = {}
                    tweet_data['created_at'] = json_tweet['created_at']
                    tweet_data['text'] = json_tweet['text']
                    json_data = json.dumps(tweet_data)
                    self.file.write(json_data.strip() + "\n")
                    self.file.flush()
                    # if it's a new day, then we zip the file saved yesterday.
                    if self.today !=  datetime.date.today():
                        self.file.close()
                        path = self.path + str(self.today)
                        subprocess.Popen(["bzip2", path])
                        # update the time to today
                        self.today = datetime.date.today()
                        # open a new file to save the data.
                        self.file = codecs.open(self.path+str(self.today), 'a', "utf-8")
                    break
        return True
        
    def on_error(self, status):
        print('on_error:')
        print (status)
        time.sleep(15*60)
        return True
if __name__ == "__main__":
############################################################################
#main code for crawling data.
#	arg[1]: keywords file
#	arg[2]: data path
#	arg[3]: twitter credentity config file
#	arg[4]: location config file
#	arg[5]: language config file
#
############################################################################
    track_list = []
    # the first arg is the keywords file.
    for line in codecs.open(sys.argv[1], 'r', "utf-8"):
        track_list.append(line.strip())
    locationBox = []
    # the fouth arg is the location config file
    if len(sys.argv) > 4:
        for line in open(sys.argv[4]):
            locationBox.append(float(line.strip()))
    language = []
    # the fifth arg is the language config file
    if len(sys.argv) > 5:
        language.append(sys.argv[5])
    else:
        language.append("en")
    #MY_TWITTER_CREDS = os.path.expanduser('~/.my_app_credentials2')
    # the third arg is the config file for twitter credentity
    MY_TWITTER_CREDS = os.path.expanduser(sys.argv[3])
    cred = open(MY_TWITTER_CREDS)
    CONSUMER_KEY = cred.readline().strip()
    CONSUMER_SECRET = cred.readline().strip()
    oauth_token = cred.readline().strip()
    oauth_secret = cred.readline().strip()
    cred.close()
    while True:
        try:
            l = StdOutListener(sys.argv[2], track_list)
            auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
            auth.set_access_token(oauth_token,  oauth_secret)
            stream = Stream(auth, l)
            if len(locationBox) > 0 and len(track_list) > 0:
                print ("filter on track, location and language for " +  sys.argv[2])
                stream.filter(track=track_list, locations=locationBox, languages=language)
            elif len(locationBox) == 0 and len(track_list) > 0:
                print ("filter on track and language for " +  sys.argv[2])
                stream.filter(track=track_list, languages=language)
            elif len(locationBox) > 0 and len(track_list) == 0:
                print ("filter on location and language for " +  sys.argv[2])
                stream.filter(locations=locationBox, languages=language) 
            else:
                print ("filter on language for " +  sys.argv[2])
                stream.sample(languages=language)
        except:
            print ("Error on path " + sys.argv[2])
            traceback.print_exc(file=sys.stdout)
            continue     
    
