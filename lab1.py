import sys
import nltk
import math
import time
import collections
import itertools
import re

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
UD_DATA_PATH = 'ud_data/'
UD_OUTPUT_PATH = 'ud_output/'


# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    
    words_count = []
    tags_count = []
    words_count.append(START_SYMBOL)
    tags_count.append(START_SYMBOL)
    for line in brown_train:
        lineWords = line.split()
        for wordWithTag in lineWords:
            wordAndTag = wordWithTag.split("/")
            if len(wordAndTag) == 2:
                words_count.append(wordAndTag[0])
                tags_count.append(wordAndTag[1])
            else:
                str = wordAndTag[0]
                for i in range(1, len(wordAndTag) - 1):
                    str = str + "/"
                    str = str + wordAndTag[i]
                tags_count.append(wordAndTag[len(wordAndTag) - 1])
                words_count.append(str)
    tags_count.append(STOP_SYMBOL)
    words_count.append(STOP_SYMBOL)
    
    brown_words.append(words_count)
    brown_tags.append(tags_count)
    return brown_words, brown_tags


# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    
    bigram_count = collections.defaultdict(int)
    trigram_count = collections.defaultdict(int)

    for sentence in brown_tags:
        for bigram in nltk.bigrams(sentence):
            bigram_count[bigram] += 1

    for sentence in brown_tags:
        for trigram in nltk.trigrams(sentence):
            trigram_count[trigram] += 1
    q_values = {k: math.log(float(v) / bigram_count[k[:2]], 2) for k, v in trigram_count.items()}
    
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()

    # trigrams.sort()
    trigrams = sorted(trigrams)
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    words_c = collections.defaultdict(int)

    for sentence in brown_words:
        for word in sentence:
            words_c[word] += 1

    for word, c in words_c.items():
        if c > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = brown_words
    for i, sentence in enumerate(brown_words):
        for j, word in enumerate(sentence):
            if word not in known_words:
                brown_words_rare[i][j] = RARE_SYMBOL
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    
    e_values_c = collections.defaultdict(int)
    tags_c = collections.defaultdict(int)

    for word_sentence, tag_sentence in zip(brown_words_rare, brown_tags):
        for word, tag in zip(word_sentence, tag_sentence):
            e_values_c[(word, tag)] += 1
            tags_c[tag] += 1

    for (word, tag), p in e_values_c.items():
        e_values[(word, tag)] = math.log(float(p) / tags_c[tag], 2)
    taglist = set(tags_c)
    
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    # emissions.sort()  for python 2
    emissions = sorted(emissions)  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    pi = collections.defaultdict(float)
    bp = {}
    bp[(-1, START_SYMBOL, START_SYMBOL)] = START_SYMBOL
    pi[(-1, START_SYMBOL, START_SYMBOL)] = 0.0

    for tokens_orig in brown_dev_words:
        tokens = [w if w in known_words else RARE_SYMBOL for w in tokens_orig]
        for w in taglist:
            if len(tokens) == 0:
                continue
            word_tag = (tokens[0], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            pi[(0, START_SYMBOL, w)] = pi[(-1, START_SYMBOL, START_SYMBOL)] + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
            bp[(0, START_SYMBOL, w)] = START_SYMBOL

        for w in taglist:
            for u in taglist:
                if len(tokens) < 2:
                    continue
                word_tag = (tokens[1], u)
                trigram = (START_SYMBOL, w, u)
                pi[(1, w, u)] = pi.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
                bp[(1, w, u)] = START_SYMBOL

        for k in range(2, len(tokens)):
            for u in taglist:
                for v in taglist:
                    max_prob = float('-Inf')
                    max_tag = ''
                    for w in taglist:
                        score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + e_values.get((tokens[k], v), LOG_PROB_OF_ZERO)
                        if (score > max_prob):
                            max_prob = score
                            max_tag = w
                    bp[(k, u, v)] = max_tag
                    pi[(k, u, v)] = max_prob

        max_prob = float('-Inf')
        v_max, u_max = None, None
        
        for (u, v) in itertools.product(taglist, taglist):
            score = pi.get((len(tokens_orig) - 1, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if score > max_prob:
                max_prob = score
                u_max = u
                v_max = v
                
        tags = []
        tags.append(v_max)
        tags.append(u_max)

        for count, k in enumerate(range(len(tokens_orig) - 3, -1, -1)):
            tags.append(bp[(k + 2, tags[count + 1], tags[count])])

        tagged_sentence = []
        tags.reverse()

        for k in range(0, len(tokens_orig)):
            tagged_sentence += [tokens_orig[k], "/", str(tags[k]), " "]
        tagged_sentence.append('\n')
        tagged.append(''.join(tagged_sentence))

    return tagged


# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

def pos(tagged, right_tegged):
    infile = open(tagged, "r")
    user_sentences = infile.readlines()
    infile.close()

    infile = open(right_tegged, "r")
    correct_sentences = infile.readlines()
    infile.close()

    num_correct = 0
    total = 0

    for user_sent, correct_sent in zip(user_sentences, correct_sentences):
        user_tok = user_sent.split()
        correct_tok = correct_sent.split()

        if len(user_tok) != len(correct_tok):
            continue

        for u, c in zip(user_tok, correct_tok):
            if u == c:
                num_correct += 1
            total += 1

    score = float(num_correct) / total * 100

    print("Percent correct tags:", score)
	
	
	
def part_1():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # print total time to run Part B
    print("Part B time: ",str(time.clock()),' sec')
	
	
	
	
def parse(input_filename, output_filename, with_tags=True):
    file_r = open(input_filename, encoding='utf-8')
    file_a = open(output_filename, 'a', encoding='utf-8')

    temp = file_r.readline()
    text_pattern = '# sentence-text: '
    word_tag_pattern = '^\\d+\\s+([\\w\\.,’:-]+)\\s+[\\w\\.,’:-]+\\s+([\\w]+)'
    while temp:
        if len(re.findall(text_pattern, temp)) > 0:
            output_string = ''
            while True:
                temp = file_r.readline()
                result = re.findall(word_tag_pattern, temp)
                if len(result) > 0:
                    if with_tags:
                        output_string += result[0][0] + '/' + result[0][1] + ' '
                    else:
                        output_string += result[0][0] + ' '
                else:
                    if temp == '\n':
                        output_string += '\n'
                        file_a.write(output_string)

                        break

        temp = file_r.readline()

    file_r.close()
    file_a.close()
	
	
	
def part_2():
    # start timer
    time.clock()
    
    parsedData = parse(UD_DATA_PATH + "en-ud-train.conllu", UD_DATA_PATH + "en-ud-train.txt")
    parsedData = parse(UD_DATA_PATH + "en-ud-dev.conllu", UD_DATA_PATH + "en-ud-tagged-dev.txt")
    parsedData = parse(UD_DATA_PATH + "en-ud-dev.conllu", UD_DATA_PATH + "en-ud-dev.txt", False)
    
    # open Brown training data
    infile = open(UD_DATA_PATH + "en-ud-train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)
    
    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, UD_OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, UD_OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, UD_OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(UD_DATA_PATH + "en-ud-dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, UD_OUTPUT_PATH + 'B5.txt')

    # print total time to run Part B
    print("Part B time: ",str(time.clock()),' sec')
	
	
part_2()
pos(UD_OUTPUT_PATH + "B5.txt", UD_DATA_PATH + "en-ud-tagged-dev.txt")


def crf(brown_words_rare, brown_tags, brown_dev_words):
    tagged = []
    ct = nltk.CRFTagger()
    train_data = []
    for i in range(len(brown_words_rare)):
        train_data.append([])
        for j in range(len(brown_words_rare[i])):
            train_data[i].append((brown_words_rare[i][j], brown_tags[i][j]))
    ct.train(train_data, 'temp.dat')

    # ct = nltk.CRFTagger()
    # ct.set_model_file('temp.dat')
    tagged_sentences = ct.tag_sents(brown_dev_words)
    for sentence in tagged_sentences:
        tagged_sentence = []
        for word_tag_pair in sentence:
            tagged_sentence += [word_tag_pair[0], "/", word_tag_pair[1], " "]
        tagged_sentence.append('\n')
        tagged.append(''.join(tagged_sentence))
    return tagged
	
	
def part_3():
    # start timer
    time.clock()
    
    parsedData = parse(UD_DATA_PATH + "en-ud-train.conllu", UD_DATA_PATH + "en-ud-train.txt")
    parsedData = parse(UD_DATA_PATH + "en-ud-dev.conllu", UD_DATA_PATH + "en-ud-tagged-dev.txt")
    parsedData = parse(UD_DATA_PATH + "en-ud-dev.conllu", UD_DATA_PATH + "en-ud-dev.txt", False)
    
    # open Brown training data
    infile = open(UD_DATA_PATH + "en-ud-train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    print('Calculating trigrams')
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, UD_OUTPUT_PATH + '/B2_crf.txt')

    # calculate list of words with count > 5 (question 3)
    print('Calculating known words')
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    print('Replacing rare words')
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, UD_OUTPUT_PATH + "/B3_crf.txt")

    # calculate emission probabilities (question 4)
    print('Calculating emission probabilities')
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, UD_OUTPUT_PATH  + "/B4_crf.txt")

    # delete unneceessary data
    del brown_train

    # open Brown development data (question 5)
    infile = open(UD_DATA_PATH + "en-ud-dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])


    print('CRF processing')
    crf_tagged = crf(brown_words_rare, brown_tags, brown_dev_words)
    q5_output(crf_tagged, UD_OUTPUT_PATH + corpus + '/B5_crf.txt')

    # print total time to run Part B
    print("Part B time: ", str(time.clock()), ' sec')

	
part_3()
pos(UD_OUTPUT_PATH + "B5_crf.txt", UD_DATA_PATH + "en-ud-tagged-dev.txt")

