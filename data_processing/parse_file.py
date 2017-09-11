import json
import re

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


stemmer = SnowballStemmer("english")
def preprocess_sentence(sentence):
    """
    Given a sentence, tokenize it, stem it, then rejoin
    it into a string.
    """
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', ' ', sentence)

    sentence = sentence.replace("\n", " __new__ ")
    sentence = sentence.replace("\t", " __tab__ ")
   
    words = [word for word in word_tokenize(sentence)]
    return "__som__ " + ' '.join([
        stemmer.stem(word) for word in words
    ]) + " __eom__"

def parse_file(filename):
  messages = []
  
  for i,line in enumerate(open(filename).readlines()):
    if i % 10000 == 0:
      print(i)

    msg = json.loads(line)
  
    if 'text' not in msg:
      continue
  
    if 'username' not in msg['from']:
      username = msg['from']['print_name']
    else:
      username = msg['from']['username']
    
    text = preprocess_sentence(msg['text'])
    date = msg['date']
  
    messages.append((username, text, date))

  return messages


messages = sorted(parse_file("chat.jsonl"), key=lambda t: t[2])

open("messages_dump.tsv", "w+").writelines([msg[0]+"\t"+msg[1]+"\t"+str(msg[2])+"\n" for msg in messages])
