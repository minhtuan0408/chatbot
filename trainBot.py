import json

from mynltk import tokenize, bag_of_words

with open ('script.json', 'r',encoding='utf-8') as f:
    script = json.load(f) 


all_words =[]
tags =[]
xy =[]

#Đọc và lấy tag, patterns trong file json
for intent in script['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        #tách từ trong pattern
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

#Bỏ các kí tự
ignore_words = ['?', '!', '.', ',']


#Tranning data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
