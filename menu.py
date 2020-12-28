import pickle, random, nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.chunk import ne_chunk

try:
    classifier_reader = open('file.pickle', 'rb')
    classifier = pickle.load(classifier_reader)
    classifier_reader.close()
except:
    pos_filename = 'positive.txt'
    neg_filename = 'negative.txt'

    pos_reader = str(open(pos_filename, 'rb'))
    neg_reader = str(open(neg_filename, 'rb'))

    pos_sentences = sent_tokenize(pos_reader)
    neg_sentences = sent_tokenize(neg_reader)

    all_words = []
    for sentence in pos_sentences:
        for word in word_tokenize(sentence):
            all_words.append(word)
    for sentence in neg_sentences:
        for word in word_tokenize(sentence):
            all_words.append(word)
    
    documents = []
    for sentence in pos_sentences:
        documents.append( (word_tokenize(sentence), 'pos') )
    for sentence in neg_sentences:
        documents.append( (word_tokenize(sentence), 'neg') )
    
    random.shuffle(documents)

    fd = FreqDist(all_words)
    word_features = list(fd.keys())[:2000]

    features_set = []
    for words, category in documents:
        words_set = set(words)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        features_set.append((features, category))
    
    train_set = features_set[:250]
    test_set = features_set[250:]

    classifier_writer = open('file.pickle', 'wb')
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    pickle.dump(classifier, classifier_writer)
    classifier_writer.close()

opinions = []

def menu():
    print('Opinion List')
    print('=============================')

    if len(opinions) == 0:
        print('No opinions')
    else:
        for i in range(len(opinions)):
            print('{}. {}'.format(i+1, opinions[i]))

    
    print('=============================\n\n')

    print('1. Insert Opinion')
    print('2. Analyze Opinion')
    print('3. Exit')

choose = 0

while choose != 3:
    choose = 0
    menu()
    while choose < 1 or choose > 3:
        try:
            choose = int(input('Choose[1-3]: '))
        except:
            print('Input must be number')
            choose = 0
    
    if choose == 1:
        opinion = ""
        while len(opinion) < 5 or len(opinion) > 30:
            opinion = input('Input your opinion[5-30]: ')
            if len(opinion) < 5 or len(opinion) > 30:
                print('Input\'s length must be between 5 and 30')
        opinions.append(opinion)
        
        print('Success to add opinion')
        input()
    elif choose == 2:
        print('Opinion List')
        print('=============================')

        if len(opinions) == 0:
            print('No opinions')
        else:
            for i in range(len(opinions)):
                print('{}. {}'.format(i+1, opinions[i]))

        
        print('=============================\n\n')
        idx = 0
        while idx < 1 or idx > len(opinions):
            try:
                idx = int(input('Choose opinion[1-{}]: '.format(len(opinions))))
            except:
                print('Input must be number')
                idx = 0
        idx-=1
        analyze = opinions[idx]
        print('Processing.....\n\n\n')

        pos_count = 0
        neg_count = 0

        for word in word_tokenize(analyze):
            if classifier.classify(FreqDist(word)) == 'pos':
                pos_count += 1
            else:
                neg_count += 1
        
        if pos_count > neg_count:
            print('Positive')
        elif pos_count < neg_count:
            print('Negative')
        else:
            print('Neutral')
        
        subchoose = ''

        while subchoose != 'yes' and subchoose != 'no':
            subchoose = input('Show analysis result? [yes/no] (case sensitive): ')
        
        if subchoose == 'no':
            input()
            continue
        elif subchoose == 'yes':
            pt = pos_tag(word_tokenize(analyze))
            fd = FreqDist(word_tokenize(analyze))

            pt = dict(pt)
            fd = dict(fd.most_common())

            for word in word_tokenize(analyze):
                synset = wordnet.synsets(word)[0]
                lemma = synset.lemmas()[0]
                synonym = lemma.name()
                antonym = '-'
                if lemma.antonyms():
                    antonym = lemma.antonyms()[0].name()
                print('{}\t{}\t{}\t{}'.format(word, pt[word], antonym, str(fd[word])))
            
            ner = ne_chunk(pos_tag(word_tokenize(analyze)))
            ner.draw()

        input()
    elif choose == 3:
        classifier_writer = open('file.pickle', 'wb')
        pickle.dump(classifier, classifier_writer)
        classifier_writer.close()