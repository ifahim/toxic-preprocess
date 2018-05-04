import Levenshtein
import string
import re


#---------------------------------------
def convert_to_lower_from_dict(word_count_list):
    new_dict = {}
    for item in word_count_list:
        new_dict[item[0]] = item[0].lower()
    return new_dict

#---------------------------------------
def remove_white_spaces_a_comment(mystr):
    """
    Remove all the whitespaces in a comment
    """
    mystr = ' '.join([x.strip() for x in mystr.split()])
    return mystr

def remove_white_spaces_from_dict(word_count_list):
    new_dict = {}
    for item in word_count_list:
        new_dict[item[0]] = remove_white_spaces_a_comment(item[0])
    return new_dict

#---------------------------------------
def remove_leaky_information(comment):
    """
    #Remove Leaky information. Ip address and user_id
    """
    comment=re.sub("\[\[User.*\]\]"," userid ",comment)
    comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}"," ipaddress ",comment)
    return comment

def remove_leaky_information_from_dict(word_count_list):
    new_dict = {}
    for item in word_count_list:
        new_dict[item[0]] = remove_leaky_information(item[0])
    return new_dict


#---------------------------------------
def extract_info_from_url(word_count_list, extractor):
    i = 0
    new_dict = {}
    for item in word_count_list:
        i += 1
        if i%50000 == 0:
            print ("Done ", i)
        word, count = item
        if extractor.has_urls(word):
            new_word = re.sub('[^0-9a-zA-Z]+', ' ', word)
            #print(item, "---------------> ", new_word)
            new_dict[word] = new_word
        else:
            new_dict[word] = word
    return new_dict

#----------------------------------------
def trim_words_len(myword_list, max_len):
    new_dict = {}
    for myword, count in myword_list:
        new_word = myword[:max_len]
        new_dict[myword] = new_word
    return new_dict

#---------------------------------------
# Replace abbreviated words
def replace_abbreviation_words_from_dict(word_count_list, abbrev_words_dict):
    new_dict = {}
    for word, count in word_count_list:
        if  word in abbrev_words_dict:
            new_dict[word] = abbrev_words_dict[word]
        else: 
            new_dict[word] = word
    return new_dict

#---------------------------------------
def strip_non_printable_chars(myStr):
    filtered_string = ''.join(filter(lambda x:x in string.printable, str(myStr)))
    return(filtered_string)

def strip_non_printable_chars_from_dict(word_count_list):
    new_dict = {}
    for item in word_count_list:
        new_dict[item[0]] = strip_non_printable_chars(item[0])
    return new_dict

#---------------------------------------
def replace_acronyms_from_dict(word_count_list, acronyms_dict, proper_words_with_profane_dict, print_out = 1):
    new_dict = {}
    for word, freq in word_count_list:
        if ((word not in proper_words_with_profane_dict) and (word in acronyms_dict)):
            new_dict[word] = acronyms_dict[word]
            if print_out == 1:
                #print (word, "--> ", acronyms_dict[word])
                continue
        else:
            new_dict[word] = word
    return new_dict

#---------------------------------------
def remove_stopwords_from_dict(word_count_list, stop_words_dict):
    new_dict = {}
    for word, count in word_count_list:
        if word in stop_words_dict:
            new_dict[word] = ''
            #print (item[0], " ====> ", '')
        else:
            new_dict[word] = word
    return new_dict

#---------------------------------------
def remove_rare_words_from_dict(word_count_list, word_dist_dict,
                                rare_word_thresh, print_out = 0):
    new_dict = {}
    for word, cnt in word_count_list:
        if  word in word_dist_dict:
            count = word_dist_dict[word]
            if count <= rare_word_thresh :
                new_dict[word] = ''
                if print_out == 1:
                    print(word, ',', cnt)
            else:
                new_dict[word] = word
        else:
            new_dict[word] = word
    return new_dict


#---------------------------------------
def remove_non_alphanumeric_from_dict(word_count_list):
    regex = re.compile('[^a-zA-Z0-9]')
    new_dict = {}
    for item in word_count_list:
        mystr = item[0]
        new_str = regex.sub('', mystr)
        new_dict[item[0]] = new_str
    return new_dict


#---------------------------------------
def remove_non_alphabet_words(word_count_list):
    regex = re.compile('[^a-zA-Z]')
    new_dict = {}
    for item in word_count_list:
        mystr = item[0]
        new_str = regex.sub('', mystr)
        new_dict[item[0]] = new_str
    return new_dict


#---------------------------------------
def remove_words_containing_non_alphabets_from_dict(word_count_list):
    new_dict = {}
    for item in word_count_list:
        mystr = item[0]
        if mystr.isalpha():
            new_dict[item[0]] = item[0]
        else:
            new_dict[item[0]] = ''
    return new_dict


#---------------------------------------
def replace_all_non_alphanumeric_chars(mystr):
    regex = re.compile('[^a-zA-Z0-9]')
    return regex.sub('', mystr)

def delete_all_non_alphabet_chars(mystr):
    regex = re.compile('[^a-zA-Z]')
    return regex.sub('', mystr)

def add_punct_in_the_end_of_a_word(orig_word, new_word):
    if orig_word[-1] in ['.', '!', ',', '?'] and new_word[-1] != '.':
        return new_word + orig_word[-1]
    else:
        return new_word

def create_regex_for_exact_word(myword):
    regex = '(?i)'
    regex = regex + '[' +  myword[0] + ']'
    for c in myword[1:]:
        regex = regex + '[' +  c + '*]{1,}'
    regex = regex + '(?:[i*][n*][g*]|[e*][d*])?'
    return regex

def check_if_profane(myword, all_badlist):
    myword = myword.lower()
    if myword in all_badlist:
        return 1, myword
    myword = replace_all_non_alphanumeric_chars(myword)
    if myword in all_badlist:
        return 1, myword
    myword = delete_all_non_alphabet_chars(myword)
    if myword in all_badlist:
        return 1, myword
    return 0, ''

def black_listed_words_regex_mapping_from_dict(word_count_list,
                                               all_badlist, profane_list_map,
                                               extreme_profane):
    new_dict = {}
    i = 0
    for myword_tup in word_count_list:
        i += 1
        if i % 50000 == 0:
            print ("Done ", i)
        myword, count = myword_tup
        f = 0
        f, new_word = check_if_profane(myword, all_badlist)
        if f == 1:
            if new_word in profane_list_map:
                new_word = profane_list_map[new_word]
            new_word = add_punct_in_the_end_of_a_word(myword, new_word)
            new_dict[myword] = new_word
            # print (myword, " ===> ", new_word)
            continue
        found = 0
        for x_word in extreme_profane:
            regex = create_regex_for_exact_word(x_word)
            p = re.compile(regex)
            if bool(p.match(myword)):
                if (((myword[0] == x_word[0]) and ('*' in myword) and (
                    abs(len(x_word) - len(myword)) <= 1)) or
                        ((x_word[0:2] == myword[0:2] and x_word[-1] == myword[
                            -1]) or
                             ((x_word[0:2] == myword[0:2] and len(
                                 x_word) == len(myword))))):
                    new_word = add_punct_in_the_end_of_a_word(myword,
                                                              x_word)
                    new_dict[myword] = new_word
                    # print(myword, ' ===> ', new_word)
                    found = 1
                    break
            elif bool(p.match(replace_all_non_alphanumeric_chars(myword))):
                myword = replace_all_non_alphanumeric_chars(myword)
                if ((myword[0] == x_word[0]) and ('*' in myword) and (
                    abs(len(x_word) - len(myword)) <= 1)):
                    new_word = add_punct_in_the_end_of_a_word(myword,
                                                              x_word)
                    new_dict[myword] = new_word
                    # print(myword, ' ===> ', new_word)
                    found = 1
                    break
            elif bool(p.match(delete_all_non_alphabet_chars(myword))):
                myword = delete_all_non_alphabet_chars(myword)
                if ((myword[0] == x_word[0]) and ('*' in myword) and (
                    abs(len(x_word) - len(myword)) <= 1)):
                    new_word = add_punct_in_the_end_of_a_word(myword,
                                                              x_word)
                    new_dict[myword] = new_word
                    # print(myword, ' ===> ', new_word)
                    found = 1
                    break

            else:
                continue
        if found == 0:
            new_dict[myword] = myword
        else:
            continue
    return new_dict


#---------------------------------------
def get_the_best_corresponding_word_using_fuzzy(myword, search_lists, matching_pct):
    dist = 0
    found = 0
    retword = ''
    for sl in search_lists:
        new_dist = Levenshtein.ratio(myword, str(sl)) #greater is better. 1 is perfect match
        if new_dist > dist:
            retword = sl
            dist = new_dist
            continue
    if (dist >= matching_pct): #Matching percent
        new_word = retword
        found = 1
        return found, new_word
    else:
        return found, myword

def replace_profane_words_using_fuzzy(word_count_list, proper_words_dict, extreme_profane, profane_list_map, badlist_common):
    new_dict = {}
    i = 0
    for item in word_count_list:
        i += 1
        if i%100000 == 0:
            print ("Done ", i)
        myword, count = item
        if (count > 500):
            new_dict[myword] = myword
            continue
        if myword in proper_words_dict:
            new_dict[myword] = myword
            continue
        found, new_word = get_the_best_corresponding_word_using_fuzzy(myword, extreme_profane, 0.8)
        if found == 1:
            if new_word in profane_list_map:
                new_word = profane_list_map[new_word]
            new_word = add_punct_in_the_end_of_a_word(myword, new_word)
            new_dict[myword] = new_word
            #print (myword, " ---> ", new_word)
            continue
        myword_1 = delete_all_non_alphabet_chars(myword)
        if myword_1 in proper_words_dict:
            new_dict[myword] = myword_1
            #print (myword, " ---> ", myword_1)
            continue
        found, new_word = get_the_best_corresponding_word_using_fuzzy(
            myword_1, badlist_common, 0.95)
        if found == 1:
            if new_word in profane_list_map:
                new_word = profane_list_map[new_word]
            new_word = add_punct_in_the_end_of_a_word(myword, new_word)
            new_dict[myword] = new_word
            #print (myword, " ---> ", new_word)
            continue
        else:
            new_dict[myword] = myword
            #print (myword, " ---> ", myword)
    return new_dict



#---------------------------------------
def check_if_proper_name(myword, citynames_dict, countries, nationalities, ethnicities, person_names_dict):
    if myword in citynames_dict:
        return 1, myword, 'cityname'
    if myword in countries:
        return 1, myword, 'country'
    if myword in nationalities:
        return 1, myword, 'nationality'
    if myword in ethnicities:
        return 1, myword, 'ethnicity'
    if myword in person_names_dict:
        return 1, myword, 'personname'
    return 0, '', ''

def check_if_proper_name_place_or_ethnicity_from_dict(word_count_list,  \
                                                      proper_words_dict, citynames_dict, countries, nationalities, \
                                                      ethnicities, person_names_dict, abstract_val = 0):
    new_dict = {}
    i = 0
    for myword_tup in word_count_list:
        i += 1
        if i%100000 == 0:
            print ("Done ", i)
        myword_orig, count = myword_tup
        myword = myword_orig
        # myword = delete_all_non_alphabet_chars(myword_orig)
        # if myword in proper_words_dict:
        #     new_dict[myword_orig] = myword
        #     continue
        found = 0
        found, new_word, abstract_value = check_if_proper_name(myword, citynames_dict, countries, nationalities, ethnicities, person_names_dict)
        if found == 1:
            if abstract_val == 0:
                new_word_final = new_word
            else:
                new_word_final = abstract_value
            new_word_final = add_punct_in_the_end_of_a_word(myword_orig, new_word_final)
            new_dict[myword_orig] = new_word_final
            #print (myword_orig, " ===> ", new_word_final)
            continue
        else:
            new_dict[myword_orig] = myword_orig
    return new_dict


#---------------------------------------
def uniq_list_preserve_order(mywords):
    output = []
    for x in mywords:
        if x not in output:
            output.append(x)
    return output

def replace_common_words_using_fuzzy(myword_list, word_dist_dict_most_common, wordnet_lemmatizer, proper_words_dict):
    i = 0
    new_dict = {}
    common_words = [x[0] for x in word_dist_dict_most_common if x[1] > 100]
    common_words = [replace_all_non_alphanumeric_chars(x) for x in common_words]
    common_words = uniq_list_preserve_order(common_words)
    common_words_dict = dict(zip(common_words, [1]*len(common_words)))

    for item in myword_list:
        i += 1
        if i%50000 == 0:
            print ("Done ", i)
        word_orig = item[0]
        if len(word_orig) > 30:
            new_dict[word_orig] = word_orig
            continue
        word = word_orig
        if word== '':
            new_dict[word_orig] = ''
            continue
        if word in common_words_dict:
            new_word = add_punct_in_the_end_of_a_word(word_orig, word)
            new_dict[word_orig] = new_word
            #print (word_orig, " ---> ", new_word)
            continue
#         lemmatized_word = wordnet_lemmatizer.lemmatize(word)
#         if  ((lemmatized_word in common_words_dict) or (lemmatized_word in proper_words_dict)):
#             new_word = add_punct_in_the_end_of_a_word(word_orig, lemmatized_word)
#             new_dict[word_orig] = new_word
#             #print (item, " ---> ", new_word)
#             continue
#         if delete_all_non_alphabet_chars(word) in proper_words_dict:
#             new_word = delete_all_non_alphabet_chars(word)
#             if len(new_word) > 3:
#                 new_word = add_punct_in_the_end_of_a_word(word_orig, new_word)
#                 new_dict[word_orig] = new_word
#                 print (word_orig, " ---> ", new_word)
#                 continue
#         word = wordnet_lemmatizer.lemmatize(word)
        matching_pct = 1 - len(word)/50.
        found, new_word = get_the_best_corresponding_word_using_fuzzy(word, common_words, matching_pct)
        if found == 1 and new_word == '':
            new_dict[word_orig] = new_word
        if found == 1 and new_word[0] == word[0]:
            new_word = add_punct_in_the_end_of_a_word(word_orig, new_word)
            new_dict[word_orig] = new_word
            print (item, " ---> ", new_word)
            continue
        else:
            new_dict[word_orig] = word_orig
    return new_dict




#--------------------------------------
def lemmatize_english_words(myword_list, wordnet_lemmatizer):
    i = 0
    new_dict = {}
    for item in myword_list:
        i += 1
        if i%50000 == 0:
            print ("Done ", i)
        myword, count = item
        if len(myword) > 30:
            new_dict[myword] = myword
            continue
        lemmatized_word = wordnet_lemmatizer.lemmatize(myword)
        new_word = add_punct_in_the_end_of_a_word(myword, lemmatized_word)
        new_dict[myword] = new_word
        #print (item, " ---> ", new_word)
        continue
    return new_dict


#--------------------------------------
def stemming_english_words(myword_list, stemmer):
    i = 0
    new_dict = {}
    for item in myword_list:
        i += 1
        if i%50000 == 0:
            print ("Done ", i)
        myword, count = item
        if len(myword) > 30:
            new_dict[myword] = myword
            continue
        stemmed_word = stemmer.stem(myword)
        new_word = add_punct_in_the_end_of_a_word(myword, stemmed_word)
        new_dict[myword] = new_word
        #print (item, " ---> ", new_word)
        continue
    return new_dict

#-------------------------------------------------

def replace_words_from_a_mapping_no_check(mystr, mapping_dict, print_out = 1):
    words = str(mystr).split()
    replaced_words = []
    for word in words:
        if word in mapping_dict:
            new_word = mapping_dict[str(word)]
            replaced_words.append(new_word)
        else:
            replaced_words.append(word)
    replaced_str = ' '.join(replaced_words)
    return replaced_str

def update_dict_with_next_level_val(new_mapping, tmp_map):
    tdict = {}
    for item in new_mapping:
        if new_mapping[item] in tmp_map:
            val = tmp_map[new_mapping[item]]
        else:
            val = new_mapping[item]
        tdict[item] = val
    return tdict