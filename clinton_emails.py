def number_of_people(sentence):
    active_sequence = False
    count = 0
    for tag in sentence.ner_tags:
        if tag == 'PERSON' and not active_sequence:
            active_sequence = True
            count += 1
        elif tag != 'PERSON' and active_sequence:
            active_sequence = False
    return count
    

def sent_has_org(sentence):
    for tag in sentence.ner_tags:
        if tag=='ORGANIZATION':
            return True
    return False
        

import re
from snorkel.lf_helpers import get_left_tokens, get_right_tokens, get_between_tokens, get_text_between
    
spouses = {'wife', 'husband', 'ex-wife', 'ex-husband'}
family = {'father', 'mother', 'sister', 'brother', 'son', 'daughter',
              'grandfather', 'grandmother', 'uncle', 'aunt', 'cousin'}
family = family | {f + '-in-law' for f in family}
other = {'boyfriend', 'girlfriend' 'boss', 'employee', 'secretary', 'co-worker'}


titles={'Representative','Assistant','Special Assistant','diplomat','official','government official','AMBASSADOR','Chancellor','Sen','Senator','Congresswoman','Congressman','Chief of Staff','mayor','Chairman','Attorney General','General',' Gen','Vice President','VP','President','dictator','Secretary','Assistant Secretary','Defense Secretary','Secretary of State','Secretary General','Gov.','Governer','Speaker','House Speaker','Democrat','Republican','PM','Minister','foreign minister','Prime Minister','ambassador','amb','Founder','Co-Founder','Author','chief executive','CEO','head of','editor','reporter','publisher','anchor','adviser','Chairman','chairwoman','chair','Rep.','columnist','leader','militant','director','deputy director','Executive Director','professor','Navy SEAL','talk show host','activist','specialist'}

def LF_political_title(c):
    return 1 if len(titles.intersection(set(get_between_tokens(c)))) > 0 else 0



def LF_title_left_window(c):
    if len(titles.intersection(set(get_left_tokens(c[0], window=2)))) > 0:
        return 1
    else:
        return 0
        
def LF_title_right_window(c):
    if len(titles.intersection(set(get_right_tokens(c[0], window=2)))) > 0:
        return 1
    else:
        return 0

def LF_no_title_in_sentence(c):
    return -1 if len(titles.intersection(set(c[0].parent.words))) == 0 else 0


LFs = [LF_political_title, LF_title_left_window, LF_title_right_window, LF_no_title_in_sentence]


from snorkel import SnorkelSession
session = SnorkelSession()
import os

from snorkel.parser import TSVDocParser
doc_parser = TSVDocParser(path="data/clinton_train.tsv")

from snorkel.parser import SentenceParser

sent_parser = SentenceParser()
from snorkel.parser import CorpusParser

cp = CorpusParser(doc_parser, sent_parser)
%time corpus = cp.parse_corpus(session, "Emails Training")
session.add(corpus)
session.commit()


for name, path in [('Emails Development', 'data/clinton_dev.tsv'),
                   ('Emails Test', 'data/clinton_test.tsv')]:
    doc_parser.path=path
    %time corpus = cp.parse_corpus(session, name)
    session.commit()

sentences = set()
for document in corpus:
    for sentence in document.sentences:
        if number_of_people(sentence) < 5:
            sentences.add(sentence)





from snorkel.models import candidate_subclass

Title = candidate_subclass('Person_Org', ['person1', 'organization'])

from snorkel.candidates import Ngrams

ngrams = Ngrams(n_max=3)

from snorkel.matchers import PersonMatcher

from snorkel.matchers import OrganizationMatcher

person_matcher = PersonMatcher(longest_match_only=True)

org_matcher = OrganizationMatcher(longest_match_only=True)

from snorkel.candidates import CandidateExtractor

ce = CandidateExtractor(Title, [ngrams, ngrams], [person_matcher, org_matcher],
                        symmetric_relations=False, nested_relations=False, self_relations=False)
						
%time c = ce.extract(sentences, 'Emails Training Candidates', session)
print "Number of candidates:", len(c)

session.add(c)
session.commit()

for corpus_name in ['Emails Development', 'Emails Test']:
    #corpus = session.query(Corpus).filter(Corpus.name == corpus_name).one()
    sentences = set()
    for document in corpus:
        for sentence in document.sentences:
            if number_of_people(sentence) < 5:
                sentences.add(sentence)
    
    %time c = ce.extract(sentences, corpus_name + ' Candidates', session)
    session.add(c)
session.commit()

from snorkel.models import CandidateSet

train = session.query(CandidateSet).filter(CandidateSet.name == 'Emails Training Candidates').one()
dev = session.query(CandidateSet).filter(CandidateSet.name == 'Emails Development Candidates').one()

from snorkel.annotations import FeatureManager

feature_manager = FeatureManager()

%time F_train = feature_manager.create(session, c, 'Train Features')


#To load existing use ..
#%time F_train = feature_manager.load(session, train, 'Train Features')						
						
from snorkel.annotations import LabelManager

label_manager = LabelManager()

%time L_train = label_manager.create(session, c, 'LF Labels', f=LFs)
L_train

from snorkel.learning import NaiveBayes

gen_model = NaiveBayes()
gen_model.train(L_train, n_iter=1000, rate=1e-5)


gen_model.save(session, 'Generative Params')
train_marginals = gen_model.marginals(L_train)
gen_model.w

from snorkel.learning import LogReg
from snorkel.learning_utils import RandomSearch, ListParameter, RangeParameter

iter_param = ListParameter('n_iter', [250, 500, 1000, 2000])
rate_param = RangeParameter('rate', 1e-4, 1e-2, step=0.75, log_base=10)
reg_param  = RangeParameter('mu', 1e-8, 1e-2, step=1, log_base=10)

disc_model = LogReg()

%time F_dev = feature_manager.update(session, dev, 'Train Features', False)

searcher = RandomSearch(disc_model, F_train, train_marginals, 10, iter_param, rate_param, reg_param)