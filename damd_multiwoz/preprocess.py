import json,  os, re, copy, zipfile
import spacy
import ontology, utils
from collections import OrderedDict
from tqdm import tqdm
from config import global_config as cfg
from db_ops import MultiWozDB
from clean_dataset import clean_slot_values, clean_text
import pdb
from sentence_transformers import SentenceTransformer, util
import torch
import pprint
import random, copy
import time

SPECIFIC_SLOT =  { # The slot not in database's item's slot
  "restaurant": [
    "time",
    "day",
    "people"
  ],
  "hotel": [
    "stay",
    "day",
    "people",
  ],
  "train": [
    "people",
  ],
  "police": [],
  "taxi":[
    "leave", 
    "destination", 
    "departure", 
    "arrive",
    "day"],
  "attraction":[],
  "hospital":[]
}

def convert2str(value):
    '''
    convert non str object to str
    '''
    if isinstance(value, list):
        return ', '.join(map(str, value))
    elif isinstance(value, dict):
        return ', '.join(f'{k}, {v}' for k, v in value.items())
    elif isinstance(value, str):
        return value
    elif isinstance(value, int):
        return str(value)
    else:
        print(f'value {value} type {type(value)} not considered!')

def extract_dom_data(data_path):
    '''
    Input: 1 of 7 domain's processed.json file
    Output: Dict, k: slot name, v: values, arranged by json's order
    '''
    with open (data_path, 'r') as f:
        data = json.loads(f.read().lower())
    slot_dict = {}
    uncomplete_slot = {'signature', 'phone', 'introduction'}
    for item in data:
        if 'restaurant' in data_path: # complete the missing slot's value with blank str
            blank_slot = uncomplete_slot - uncomplete_slot.intersection(set(item.keys()))
            for slot in list(blank_slot):
                item[slot] = ' '
        for slot, value in item.items():
            if not slot_dict.get(slot):
                slot_dict[slot] = [convert2str(value)]
            else:
                slot_dict[slot].append(convert2str(value))
    return slot_dict

def extract_value_data(data_path, dom, slot):
    '''
    Input: 
        data_path: the value_set_processed.json path
        dom: query domain
        slot: query slot
    Output:
        list of slot's values arranged by json's order
    '''
    if dom not in SPECIFIC_SLOT.keys():
        print('There are no special info slot in dom ', dom)
        return []
    else:
        with open (data_path, 'r') as f:
            data = json.loads(f.read().lower())
            return data[dom][slot] 
def get_db_values(value_set_path):
    processed = {}
    bspn_word = []
    nlp = spacy.load('en_core_web_sm')

    with open(value_set_path, 'r') as f:
        value_set = json.loads(f.read().lower())

    with open('db/ontology.json', 'r') as f:
        otlg = json.loads(f.read().lower())

    for domain, slots in value_set.items():
        processed[domain] = {}
        bspn_word.append('['+domain+']')
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot)
            if s_p in ontology.informable_slots[domain]:
                bspn_word.append(s_p)
                processed[domain][s_p] = []

    '''
    #bspn_word
    ['[police]', '[taxi]', '[restaurant]', 'name', 'area', 'pricerange', 'food', '[attraction]', 'name', 'area', 'type', '[hotel]', 'name', 'internet', '
area', 'parking', 'stars', 'type', 'pricerange', '[hospital]', 'department', '[train]', 'departure', 'day', 'arrive', 'destination', 'leave']  
    # processed
    {'police': {}, 'taxi': {}, 'restaurant': {'name': [], 'area': [], 'pricerange': [], 'food': []}, 'attraction': {'name': [], 'area': [], 'type': []}, 
'hotel': {'name': [], 'internet': [], 'area': [], 'parking': [], 'stars': [], 'type': [], 'pricerange': []}, 'hospital': {'department': []}, 'train':
 {'departure': [], 'day': [], 'arrive': [], 'destination': [], 'leave': []}}
    '''

    # add values to processed's domain-slot and bspn_word
    for domain, slots in value_set.items():
        for slot, values in slots.items():
            s_p = ontology.normlize_slot_names.get(slot, slot) # s_p for slot preprocessed
            if s_p in ontology.informable_slots[domain]:
                for v in values:
                    _, v_p = clean_slot_values(domain, slot, v)
                    v_p = ' '.join([token.text for token in nlp(v_p)]).strip() # tokenization
                    processed[domain][s_p].append(v_p)
                    for x in v_p.split():
                        if x not in bspn_word:
                            bspn_word.append(x)
    # add potentielly omitted item in prev steps?
    for domain_slot, values in otlg.items():
        domain, slot = domain_slot.split('-')
        if domain == 'bus':
            domain = 'taxi'
        if slot == 'price range':
            slot = 'pricerange'
        if slot == 'book stay':
            slot = 'stay'
        if slot == 'book day':
            slot = 'day'
        if slot == 'book people':
            slot = 'people'
        if slot == 'book time':
            slot = 'time'
        if slot == 'arrive by':
            slot = 'arrive'
        if slot == 'leave at':
            slot = 'leave'
        if slot == 'leaveat':
            slot = 'leave'
        if slot not in processed[domain]:
            processed[domain][slot] = []
            bspn_word.append(slot)
        for v in values:
            _, v_p = clean_slot_values(domain, slot, v)
            v_p = ' '.join([token.text for token in nlp(v_p)]).strip()
            if v_p not in processed[domain][slot]:
                processed[domain][slot].append(v_p)
                for x in v_p.split():
                    if x not in bspn_word:
                        bspn_word.append(x)

    with open(value_set_path.replace('.json', '_processed.json'), 'w') as f:
        json.dump(processed, f, indent=2)
    with open('data/multi-woz-processed/bspn_word_collection.json', 'w') as f:
        json.dump(bspn_word, f, indent=2)

    print('DB value set processed! ')

def preprocess_db(db_paths):
    dbs = {}
    nlp = spacy.load('en_core_web_sm')
    for domain in ontology.all_domains:
        with open(db_paths[domain], 'r') as f:
            dbs[domain] = json.loads(f.read().lower())
            for idx, entry in enumerate(dbs[domain]):
                new_entry = copy.deepcopy(entry)
                for key, value in entry.items():
                    if type(value) is not str:
                        continue
                    del new_entry[key]
                    key, value = clean_slot_values(domain, key, value)
                    tokenize_and_back = ' '.join([token.text for token in nlp(value)]).strip()
                    new_entry[key] = tokenize_and_back
                dbs[domain][idx] = new_entry
        with open(db_paths[domain].replace('.json', '_processed.json'), 'w') as f:
            json.dump(dbs[domain], f, indent=2)
        print('[%s] DB processed! '%domain)


class DataPreprocessor(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.db = MultiWozDB(cfg.dbs)
        data_path = 'data/multi-woz/annotated_user_da_with_span_full.json'
        archive = zipfile.ZipFile(data_path + '.zip', 'r')
        self.convlab_data = json.loads(archive.open(data_path.split('/')[-1], 'r').read().lower())
        self.delex_sg_valdict_path = 'data/multi-woz-processed/delex_single_valdict.json'
        self.delex_mt_valdict_path = 'data/multi-woz-processed/delex_multi_valdict.json'
        self.ambiguous_val_path = 'data/multi-woz-processed/ambiguous_values.json'
        self.delex_refs_path = 'data/multi-woz-processed/reference_no.json'
        self.delex_refs = json.loads(open(self.delex_refs_path, 'r').read())
        if not os.path.exists(self.delex_sg_valdict_path):
            self.delex_sg_valdict, self.delex_mt_valdict, self.ambiguous_vals = self.get_delex_valdict()
        else:
            self.delex_sg_valdict = json.loads(open(self.delex_sg_valdict_path, 'r').read())
            self.delex_mt_valdict = json.loads(open(self.delex_mt_valdict_path, 'r').read())
            self.ambiguous_vals = json.loads(open(self.ambiguous_val_path, 'r').read())

        self.vocab = utils.Vocab(cfg.vocab_size)
        self.value_set_path = 'db/value_set_processed.json'
        self.pre_db_paths = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.device = 'cuda:0'
        self.model = SentenceTransformer(cfg.preprocess_model_path)


    def delex_by_annotation(self, dial_turn):
        u = dial_turn['text'].split()
        span = dial_turn['span_info']
        for s in span:
            slot = s[1]
            if slot == 'open':
                continue
            if ontology.da_abbr_to_slot_name.get(slot):
                slot = ontology.da_abbr_to_slot_name[slot]
            for idx in range(s[3], s[4]+1):
                u[idx] = ''
            try:
                u[s[3]] = '[value_'+slot+']'
            except:
                u[5] = '[value_'+slot+']'
        u_delex = ' '.join([t for t in u if t is not ''])
        u_delex = u_delex.replace('[value_address] , [value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_name] [value_name]', '[value_name]')
        u_delex = u_delex.replace('[value_name]([value_phone] )', '[value_name] ( [value_phone] )')
        return u_delex


    def delex_by_valdict(self, text):
        text = clean_text(text)

        text = re.sub(r'\d{5}\s?\d{5,7}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)', '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        text = re.sub(r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})', '[value_postcode]', text)

        for value, slot in self.delex_mt_valdict.items():
            text = text.replace(value, '[value_%s]'%slot)

        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    tokens[idx] = '[value_%s]'%slot
            text = ' '.join(tokens)

        for ambg_ent in self.ambiguous_vals:
            start_idx = text.find(' '+ambg_ent)   # ely is a place, but appears in words like moderately
            if start_idx == -1:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival', 'destination', 'there', 'reach',  'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type=='time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                                'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type=='time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        text = text.replace('[value_car] [value_car]', '[value_car]')
        return text


    def get_delex_valdict(self, ):
        skip_entry_type = {
            'taxi': ['taxi_phone'],
            'police': ['id'],
            'hospital': ['id'],
            'hotel': ['id', 'location', 'internet', 'parking', 'takesbookings', 'stars', 'price', 'n', 'postcode', 'phone'],
            'attraction': ['id', 'location', 'pricerange', 'price', 'openhours', 'postcode', 'phone'],
            'train': ['price', 'id'],
            'restaurant': ['id', 'location', 'introduction', 'signature', 'type', 'postcode', 'phone'],
        }
        entity_value_to_slot= {}
        ambiguous_entities = []
        for domain, db_data in self.db.dbs.items():
            print('Processing entity values in [%s]'%domain)
            if domain != 'taxi':
                for db_entry in db_data:
                    for slot, value in db_entry.items():
                        if slot not in skip_entry_type[domain]:
                            if type(value) is not str:
                                raise TypeError("value '%s' in domain '%s' should be rechecked"%(slot, domain))
                            else:
                                slot, value = clean_slot_values(domain, slot, value)
                                value = ' '.join([token.text for token in self.nlp(value)]).strip()
                                if value in entity_value_to_slot and entity_value_to_slot[value] != slot:
                                    # print(value, ": ",entity_value_to_slot[value], slot)
                                    ambiguous_entities.append(value)
                                entity_value_to_slot[value] = slot
            else:   # taxi db specific
                db_entry = db_data[0]
                for slot, ent_list in db_entry.items():
                    if slot not in skip_entry_type[domain]:
                        for ent in ent_list:
                            entity_value_to_slot[ent] = 'car'
        ambiguous_entities = set(ambiguous_entities)
        ambiguous_entities.remove('cambridge')
        ambiguous_entities = list(ambiguous_entities)
        for amb_ent in ambiguous_entities:   # departure or destination? arrive time or leave time?
            entity_value_to_slot.pop(amb_ent)
        entity_value_to_slot['parkside'] = 'address'
        entity_value_to_slot['parkside, cambridge'] = 'address'
        entity_value_to_slot['cambridge belfry'] = 'name'
        entity_value_to_slot['hills road'] = 'address'
        entity_value_to_slot['hills rd'] = 'address'
        entity_value_to_slot['Parkside Police Station'] = 'name'

        single_token_values = {}
        multi_token_values = {}
        for val, slt in entity_value_to_slot.items():
            if val in ['cambridge']:
                continue
            if len(val.split())>1:
                multi_token_values[val] = slt
            else:
                single_token_values[val] = slt

        with open(self.delex_sg_valdict_path, 'w') as f:
            single_token_values = OrderedDict(sorted(single_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(single_token_values, f, indent=2)
            print('single delex value dict saved!')
        with open(self.delex_mt_valdict_path, 'w') as f:
            multi_token_values = OrderedDict(sorted(multi_token_values.items(), key=lambda kv:len(kv[0]), reverse=True))
            json.dump(multi_token_values, f, indent=2)
            print('multi delex value dict saved!')
        with open(self.ambiguous_val_path, 'w') as f:
            json.dump(ambiguous_entities, f, indent=2)
            print('ambiguous value dict saved!')

        return single_token_values, multi_token_values, ambiguous_entities

    def construct_cons_random(self, cons_dict, cross_domain=False):
        '''
        choose one random item within domain, replace current constraints with the random one 

        Input:  cons_dict, OrderedDict
        Ex: [('hotel', OrderedDict([('pricerange', 'cheap'), ('type', 'hotel')]))]

        Output: cons_dict_random, OrderedDict
        '''
        # note that only system turn(metadate not None) can have constraints
        return None
    def construct_cons_max_multi(self, cons_dict, prev_cons_dict, cross_domain=False, debug=False):
        '''
        choose cfg.topk_cntfact item from 2 * cfg.topk_cntfact items 
        return cfg.topk_cntfact cntfact slot-value list. 
        select the most similar and not diff items.

        Input:  
        cons_dict, OrderedDict
        Ex: [('hotel', OrderedDict([('pricerange', 'cheap'), ('type', 'hotel')]))]
        prev_cons_dict, list of OrderedDict

        Output: list of cons_dict_random, [OrderedDict]
        '''
        if len(list(cons_dict.keys())) >=2 and debug:
            print('current cons_dict contains more than 1 dom:', list(cons_dict.keys()))
        for dom, slot_values in cons_dict.items():
            total_scores = 0
            dom_dict = extract_dom_data(self.pre_db_paths[dom])
            db_slot_num = 0
            prev_slot = prev_cons_dict[0][dom].keys() if prev_cons_dict[0].get(dom) else [] # all prev_cons_dict has same slot, diff values, 0 is ok
            special_slot = SPECIFIC_SLOT[dom] if SPECIFIC_SLOT.get(dom) else []
            top_k = cfg.topk_cntfact
            for slot, value in slot_values.items():
                if slot not in prev_slot:
                    if slot not in special_slot:
                        db_slot_num += 1 
                        query = convert2str(value)
                        corpus = dom_dict[slot]
                        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
                        query_embedding = self.model.encode(query, convert_to_tensor=True)
                        #top_k = min(cfg.topk_cntfact, len(corpus))
                        # We use cosine-similarity and torch.topk to find the highest 5 scores
                        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                        #print(f'cos_scors shape: {cos_scores.shape}')
                        total_scores += cos_scores
                    '''
                    else: # for special slot, choose one from value_set
                        db_slot_num += 1
                        query = convert2str(value)
                        corpus = extract_value_data(self.value_set_path, dom, slot)
                    '''
            if db_slot_num > 0:           
                total_scores /= db_slot_num
                domain_top_results = torch.topk(total_scores, k=2*top_k)
                top_k_score = domain_top_results[0].tolist()
                top_k_index = domain_top_results[1].tolist()
                current_item = top_k_index.pop(0)
                current_score = top_k_score.pop(0)
                selected_indices = [] 
                selected_scores = []
                for i in range(len(top_k_index)):  
                    if len(selected_indices) >= top_k:
                        break
                    if round(top_k_score[i], 4) not in selected_scores:
                        selected_scores.append(round(top_k_score[i], 4))
                        selected_indices.append(top_k_index[i])
                if len(selected_indices) < top_k: # not enough cntfact items, by default add new ones from top to bottom
                    for i in range(top_k):
                        if len(selected_indices) >= top_k:
                            break
                        if top_k_index[i] not in selected_indices:
                            selected_scores.append(round(top_k_score[i], 4))
                            selected_indices.append(top_k_index[i])
            # replace cntfact value for current cons_dict's slot
            with open (self.pre_db_paths[dom], 'r') as f:
                dom_data = json.loads(f.read().lower())
                if db_slot_num > 0:
                    cntfact_items = [dom_data[cntfact_index] for cntfact_index in selected_indices]
                    if debug:
                        print('\n =========================== \n')
                        print('cntfact item ')
                        pprint.pprint(cntfact_items)

                cntfact_cons_dict = copy.deepcopy(prev_cons_dict)
                if debug:
                    print("\n ************* ori cntfact_cons_dict **************")
                    print(cntfact_cons_dict)
                # only consider genearte multiple cntfact here, and prev_cons_dict is list of OrderedDict or None
                for i in range(len(prev_cons_dict)):
                    if not prev_cons_dict[i].get(dom):
                        #cntfact_cons_dict = copy.deepcopy(prev_cons_dict) # TODO: only copy the single dom's cons_dict, otherwise the prev processed will be modified.
                        cntfact_cons_dict[i][dom] = cons_dict[dom] # add new domain's metadata and modify it
                if debug: 
                    print("\n ************* copyed but not modified cntfact_cons_dict **************")
                    print(cntfact_cons_dict)
                for slot, value in slot_values.items():
                    if db_slot_num > 0 :
                        if slot not in prev_slot: # cntfact_item = dom_data[cntfact_indice]
                            for i, cntfact_item in enumerate(cntfact_items):
                                if cntfact_item.get(slot): # current_item = top_k_index.pop(0)
                                    cntfact_cons_dict[i][dom][slot] = cntfact_item[slot]
                                else:
                                    cntfact_cons_dict[i][dom][slot] = ' '
                    elif slot not in prev_slot and slot in SPECIFIC_SLOT[dom]:
                        query = convert2str(value)
                        corpus = extract_value_data(self.value_set_path, dom, slot)
                        top_k = min(2*cfg.topk_cntfact, len(corpus))
                        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
                        query_embedding = self.model.encode(query, convert_to_tensor=True)
                        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0] 
                        if top_k > len(corpus):
                            print(f"ATTENTION! top_cntfact got {top_k} but corpus got {len(corpus)}")
                        #pdb.set_trace()
                        slot_top_results = torch.topk(cos_scores, k=top_k)
                        top_k_index = slot_top_results[1].tolist()
                        top_k_score = slot_top_results[0].tolist()
                        current_indice = top_k_index.pop(0)
                        current_score = top_k_score.pop(0)
                        current_value = query
                        selected_indices = [] 
                        selected_scores = []
                        for i in range(len(top_k_index)):  
                            if len(selected_indices) >= cfg.topk_cntfact:
                                break
                            if round(top_k_score[i], 4) not in selected_scores:
                                selected_scores.append(round(top_k_score[i], 4))
                                selected_indices.append(top_k_index[i])
                        if len(selected_indices) < cfg.topk_cntfact: # not enough cntfact items, by default add new ones from top to bottom
                            for i in range(cfg.topk_cntfact):
                                if len(selected_indices) >= cfg.topk_cntfact:
                                    break
                                if top_k_index[i] not in selected_indices:
                                    selected_scores.appen(round(top_k_score[i], 4))
                                    selected_indices.append(top_k_index[i])
                        if debug:
                            print('****************** current special value ****************')
                            print(current_value)
                            print('****************** cntfact special value ****************')
                            print([corpus[i] for i in selected_indices])
                        for i, cntfact_index in enumerate(selected_indices):
                            cntfact_cons_dict[i][dom][slot] = corpus[cntfact_index]
                    #else:
                    #    cntfact_cons_dict[dom][slot] = ' '
            prev_cons_dict = cntfact_cons_dict
            if debug:
                print('\n ******** modified prev_cons_dict **********')
                print(prev_cons_dict)
                
            if debug:                            
                print('\n ************** current cons_dict ***************')
                print(cons_dict)
                if db_slot_num > 0:
                    print('original item:')
                    #pprint.pprint(dom_data[current_item]) 
                    print("Query:", query)
                    '''
                    print("\nTop 5 most similar sentences in corpus:")
                    print('top scores and index:')
                    for score, idx in zip(domain_top_results[0], domain_top_results[1]):
                        print("index: {}".format(idx), "(Score: {:.4f})".format(score))
                    '''
                print('\n ************ final turn cntfact_consdict *************')
                print(cntfact_cons_dict)
                
        return cntfact_cons_dict

    def construct_cons_max(self, cons_dict, prev_cons_dict, cross_domain=False, debug=False):
        '''
        choose most simliar item within domain, replace current constraints with the random one 
        

        Input:  
        cons_dict, OrderedDict
        Ex: [('hotel', OrderedDict([('pricerange', 'cheap'), ('type', 'hotel')]))]
        multiple: return cfg.topk_cntfact result. select the most similar and not diff scores 
        Output: cons_dict_random, OrderedDict
        '''
        if len(list(cons_dict.keys())) >=2 and debug:
            print('current cons_dict contains more than 1 dom:', list(cons_dict.keys()))
        for dom, slot_values in cons_dict.items():
            total_scores = 0
            dom_dict = extract_dom_data(self.pre_db_paths[dom])
            db_slot_num = 0
            prev_slot = prev_cons_dict[dom].keys() if prev_cons_dict.get(dom) else []
            special_slot = SPECIFIC_SLOT[dom] if SPECIFIC_SLOT.get(dom) else []
            top_k = cfg.topk_cntfact
            for slot, value in slot_values.items():
                if slot not in prev_slot:
                    if slot not in special_slot:
                        db_slot_num += 1 
                        query = convert2str(value)
                        corpus = dom_dict[slot]
                        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
                        query_embedding = self.model.encode(query, convert_to_tensor=True)
                        #top_k = min(cfg.topk_cntfact, len(corpus))
                        # We use cosine-similarity and torch.topk to find the highest 5 scores
                        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                        #print(f'cos_scors shape: {cos_scores.shape}')
                        total_scores += cos_scores
                    '''
                    else: # for special slot, choose one from value_set
                        db_slot_num += 1
                        query = convert2str(value)
                        corpus = extract_value_data(self.value_set_path, dom, slot)
                    '''
            if db_slot_num > 0:           
                total_scores /= db_slot_num
                domain_top_results = torch.topk(total_scores, k=top_k)
                top_k_index = domain_top_results[1].tolist()
                current_item = top_k_index.pop(0)

                cntfact_indice = random.choice(top_k_index)
            # replace cntfact value for current cons_dict's slot
            with open (self.pre_db_paths[dom], 'r') as f:
                dom_data = json.loads(f.read().lower())
                if db_slot_num > 0:
                    cntfact_item = dom_data[cntfact_indice]
                    if debug:
                        print('\n =========================== \n')
                        #print('cntfact item ')
                        #pprint.pprint(cntfact_item)

                cntfact_cons_dict = copy.deepcopy(prev_cons_dict)
                if debug:
                    print("\n ************* ori cntfact_cons_dict **************")
                    print(cntfact_cons_dict)
                if not prev_cons_dict.get(dom):
                    #cntfact_cons_dict = copy.deepcopy(prev_cons_dict) # TODO: only copy the single dom's cons_dict, otherwise the prev processed will be modified.
                    cntfact_cons_dict[dom] = cons_dict[dom] # add new domain's metadata and modify it
                if debug: 
                    print("\n ************* copyed but not modified cntfact_cons_dict **************")
                    print(cntfact_cons_dict)
                for slot, value in slot_values.items():
                    if db_slot_num > 0 :
                        if slot not in prev_slot:
                            if cntfact_item.get(slot):
                                cntfact_cons_dict[dom][slot] = cntfact_item[slot]
                            else:
                                cntfact_cons_dict[dom][slot] = ' '
                    elif slot not in prev_slot and slot in SPECIFIC_SLOT[dom]:
                        query = convert2str(value)
                        corpus = extract_value_data(self.value_set_path, dom, slot)
                        top_k = min(cfg.topk_cntfact, len(corpus))
                        corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
                        query_embedding = self.model.encode(query, convert_to_tensor=True)
                        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0] 
                        slot_top_results = torch.topk(cos_scores, k=top_k)
                        top_k_index = slot_top_results[1].tolist()
                        current_indice = top_k_index.pop(0)
                        current_value = query
                        cntfact_indice = random.choice(top_k_index)
                        if debug:
                            print('****************** current special value ****************')
                            print(current_value)
                            print('****************** cntfact special value ****************')
                            print(corpus[cntfact_indice])
                        cntfact_cons_dict[dom][slot] = corpus[cntfact_indice]
                    #else:
                    #    cntfact_cons_dict[dom][slot] = ' '
            prev_cons_dict = cntfact_cons_dict
            if debug:
                print('\n ******** modified prev_cons_dict **********')
                print(prev_cons_dict)
                
            if debug:                            
                print('\n ************** current cons_dict ***************')
                print(cons_dict)
                if db_slot_num > 0:
                    print('original item:')
                    #pprint.pprint(dom_data[current_item]) 
                    print("Query:", query)
                    '''
                    print("\nTop 5 most similar sentences in corpus:")
                    print('top scores and index:')
                    for score, idx in zip(domain_top_results[0], domain_top_results[1]):
                        print("index: {}".format(idx), "(Score: {:.4f})".format(score))
                    '''
                print('\n ************ final turn cntfact_consdict *************')
                print(cntfact_cons_dict)
                
        return cntfact_cons_dict

    def preprocess_main(self, save_path=None, is_test=False):
        """
        """
        start_time = time.time()
        cntfact_total_time = 0
        data = {}
        count=0
        self.unique_da = {}
        ordered_sysact_dict = {}
        for fn, raw_dial in tqdm(list(self.convlab_data.items())):
            count +=1
            #if count == 327:
                #pdb.set_trace()

            compressed_goal = {}
            dial_domains, dial_reqs = [], []
            for dom, g in raw_dial['goal'].items():
                if dom != 'topic' and dom != 'message' and g:
                    if g.get('reqt'):
                        for i, req_slot in enumerate(g['reqt']):
                            if ontology.normlize_slot_names.get(req_slot):
                                g['reqt'][i] = ontology.normlize_slot_names[req_slot]
                                dial_reqs.append(g['reqt'][i])
                    compressed_goal[dom] = g
                    if dom in ontology.all_domains:
                        dial_domains.append(dom)

            dial_reqs = list(set(dial_reqs))

            dial = {'goal': compressed_goal, 'log': []}
            single_turn = {}
            constraint_dict = OrderedDict()
            prev_constraint_dict = {} # origin code's sentinal
            if cfg.enable_cntfact:
                prev_cntfact_constraint_dict = OrderedDict() if not cfg.enable_multi_cntfact else [OrderedDict() for _ in range(cfg.topk_cntfact)]# initialize prev_cntfact_constraint_dict
            prev_turn_domain = ['general']
            ordered_sysact_dict[fn] = {}

            for turn_num, dial_turn in enumerate(raw_dial['log']):

                dial_state = dial_turn['metadata']
                if not dial_state:   # user
                    u = ' '.join(clean_text(dial_turn['text']).split())
                    if dial_turn['span_info']:
                        u_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        u_delex = self.delex_by_valdict(dial_turn['text'])

                    single_turn['user'] = u
                    single_turn['user_delex'] = u_delex

                else:   #system
                    if dial_turn['span_info']:
                        s_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        if not dial_turn['text']:
                            print(fn)
                        s_delex = self.delex_by_valdict(dial_turn['text'])
                    single_turn['resp'] = s_delex
                    single_turn['resp_nodelex'] = ' '.join(clean_text(dial_turn['text']).split())

                    # get belief state
                    for domain in dial_domains:
                        if not constraint_dict.get(domain):
                            constraint_dict[domain] = OrderedDict()
                        info_sv = dial_state[domain]['semi']
                        for s,v in info_sv.items():
                            s,v = clean_slot_values(domain, s,v)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v
                        book_sv = dial_state[domain]['book']
                        for s,v in book_sv.items():
                            if s == 'booked':
                                continue
                            s,v = clean_slot_values(domain, s,v)
                            if len(v.split())>1:
                                v = ' '.join([token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v
                    constraints = []
                    cons_delex = []
                    turn_dom_bs = []
                    for domain, info_slots in constraint_dict.items(): # add conuter_fact belief state here as additional dict
                        if info_slots:
                            constraints.append('['+domain+']')
                            cons_delex.append('['+domain+']')
                            for slot, value in info_slots.items():
                                constraints.append(slot)
                                constraints.extend(value.split()) # add slot and value. ex: ['[hotel]', 'pricerange', 'cheap', 'type', 'hotel']
                                cons_delex.append(slot) # add only slot. ex: ['[hotel]', 'pricerange', 'type']
                            if domain not in prev_constraint_dict:
                                turn_dom_bs.append(domain)
                            elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                turn_dom_bs.append(domain)
                    if cfg.cntfact_max_mode:
                        proba_active = random.random()
                        #print('current proba:', proba_active)
                        cntfact_active = proba_active<= cfg.cntfact_raio
                        #print('current switch', cntfact_active)
                        if cntfact_active:
                            #if count >= 327:
                                #pdb.set_trace()
                            #print('********** Outer Function **********')
                            #print('********** prev_cntfact_constraint_dict **********')
                            #print(prev_cntfact_constraint_dict)
                            #pdb.set_trace()
                            cntfact_start = time.time()
                            if cfg.enable_multi_cntfact:
                                cntfact_constraint_dict = self.construct_cons_max_multi(constraint_dict, prev_cntfact_constraint_dict, debug=False)
                            else:
                                cntfact_constraint_dict = self.construct_cons_max(constraint_dict, prev_cntfact_constraint_dict, debug=False)
                            cntfact_end = time.time()
                            cntfact_total_time += cntfact_end - cntfact_start
                            if count % 100 == 0:
                                print('current avarage time consumption', cntfact_total_time / count)
                            cntfact_constraints = []
                            cntfact_cons_delex = []
                            cntfact_turn_dom_bs = []
                            if cfg.enable_multi_cntfact:
                                for i, single_dict in enumerate(cntfact_constraint_dict):
                                    _cntfact_constraint = []
                                    _cntfact_cons_delex = []
                                    _cntfact_turn_dom_bs = []
                                    for domain, info_slots in single_dict.items(): # add conuter_fact belief state here as additional dict
                                        if info_slots:
                                            _cntfact_constraint.append('['+domain+']')
                                            _cntfact_cons_delex.append('['+domain+']')
                                            for slot, value in info_slots.items():
                                                _cntfact_constraint.append(slot)
                                                _cntfact_constraint.extend(value.split()) # add slot and value. ex: ['[hotel]', 'pricerange', 'cheap', 'type', 'hotel']
                                                _cntfact_cons_delex.append(slot) # add only slot. ex: ['[hotel]', 'pricerange', 'type']
                                            if domain not in prev_constraint_dict:
                                                _cntfact_turn_dom_bs.append(domain)
                                            elif prev_constraint_dict[domain] != cntfact_constraint_dict[i][domain]:
                                                _cntfact_turn_dom_bs.append(domain) 
                                    cntfact_constraints.append(_cntfact_constraint)
                                    cntfact_cons_delex.append(_cntfact_cons_delex)
                                    cntfact_turn_dom_bs.append(_cntfact_turn_dom_bs)
                            else:
                                for domain, info_slots in cntfact_constraint_dict.items(): # add conuter_fact belief state here as additional dict
                                    if info_slots:
                                        cntfact_constraints.append('['+domain+']')
                                        cntfact_cons_delex.append('['+domain+']')
                                        for slot, value in info_slots.items():
                                            cntfact_constraints.append(slot)
                                            cntfact_constraints.extend(value.split()) # add slot and value. ex: ['[hotel]', 'pricerange', 'cheap', 'type', 'hotel']
                                            cntfact_cons_delex.append(slot) # add only slot. ex: ['[hotel]', 'pricerange', 'type']
                                        if domain not in prev_constraint_dict:
                                            cntfact_turn_dom_bs.append(domain)
                                        elif prev_constraint_dict[domain] != cntfact_constraint_dict[domain]:
                                            cntfact_turn_dom_bs.append(domain) 
                        else:
                            if cfg.enable_multi_cntfact:
                                cntfact_constraints = [constraints for _ in range(cfg.topk_cntfact)]
                                cntfact_cons_delex = [cons_delex for _ in range(cfg.topk_cntfact)]
                                cntfact_turn_dom_bs = [turn_dom_bs for _ in range(cfg.topk_cntfact)]
                            else:
                                cntfact_constraints = constraints
                                cntfact_cons_delex = cons_delex
                                cntfact_turn_dom_bs = turn_dom_bs

                    sys_act_dict = {}
                    turn_dom_da = set()
                    for act in dial_turn['dialog_act']: # {'hotel-request': [['area', '?']]}
                        d, a = act.split('-')
                        turn_dom_da.add(d)
                    turn_dom_da = list(turn_dom_da)
                    if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                        turn_dom_da.remove('general')
                    if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                        turn_dom_da.remove('booking')

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]

                    # get system action
                    for dom in turn_domain:
                        sys_act_dict[dom] = {}
                    add_to_last_collect = []
                    booking_act_map = {'inform': 'offerbook', 'book': 'offerbooked'}
                    for act, params in dial_turn['dialog_act'].items():
                        if act == 'general-greet':
                            continue
                        d, a = act.split('-') # domain-act. ex: hotel-request
                        if d == 'general' and d not in sys_act_dict:
                            sys_act_dict[d] = {}
                        if d == 'booking':
                            d = turn_domain[0]
                            a = booking_act_map.get(a, a)
                        add_p = []
                        for param in params:
                            p = param[0]
                            if p == 'none':
                                continue
                            elif ontology.da_abbr_to_slot_name.get(p):
                                p = ontology.da_abbr_to_slot_name[p]
                            if p not in add_p:
                                add_p.append(p)
                        add_to_last = True if a in ['request', 'reqmore', 'bye', 'offerbook'] else False
                        if add_to_last:
                            add_to_last_collect.append((d,a,add_p))
                        else: # inform. ex: {'booking-inform': [['none', 'none']], 'hotel-inform': [['price', 'cheap'], ['choice', '1'], ['parking', 'none']]} -> output: sys_act_dict[d] = {'inform': ['price', 'choice', 'parking']}
                            sys_act_dict[d][a] = add_p
                    for d, a, add_p in add_to_last_collect:
                        sys_act_dict[d][a] = add_p

                    for d in copy.copy(sys_act_dict):
                        acts = sys_act_dict[d]
                        if not acts:
                            del sys_act_dict[d]
                        if 'inform' in acts and 'offerbooked' in acts:
                            for s in sys_act_dict[d]['inform']:
                                sys_act_dict[d]['offerbooked'].append(s)
                            del sys_act_dict[d]['inform']


                    ordered_sysact_dict[fn][len(dial['log'])] = sys_act_dict

                    sys_act = []
                    if 'general-greet' in dial_turn['dialog_act']:
                        sys_act.extend(['[general]', '[greet]'])
                    for d, acts in sys_act_dict.items():
                        sys_act += ['[' + d + ']']
                        for a, slots in acts.items():
                            self.unique_da[d+'-'+a] = 1
                            sys_act += ['[' + a + ']']
                            sys_act += slots


                    # get db pointers
                    matnums = self.db.get_match_num(constraint_dict)
                    match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
                    match = matnums[match_dom]
                    dbvec = self.db.addDBPointer(match_dom, match)
                    bkvec = self.db.addBookingPointer(dial_turn['dialog_act'])

                    single_turn['pointer'] = ','.join([str(d) for d in dbvec + bkvec])
                    single_turn['match'] = str(match)
                    single_turn['constraint'] = ' '.join(constraints)
                    single_turn['cons_delex'] = ' '.join(cons_delex)
                    if cfg.enable_cntfact and cfg.cntfact_max_mode:
                        if cfg.enable_multi_cntfact:
                            single_turn['cntfact_constraint_max'] = [' '.join(_cntfact_constraint) for _cntfact_constraint in cntfact_constraints]
                            single_turn['cntfact_cons_delex_max'] = [' '.join(_cntfact_cons_delex) for _cntfact_constraint in cntfact_cons_delex]
                        else:
                            single_turn['cntfact_constraint_max'] = ' '.join(cntfact_constraints)
                            single_turn['cntfact_cons_delex_max'] = ' '.join(cntfact_cons_delex)
                    single_turn['sys_act'] = ' '.join(sys_act)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(['['+d+']' for d in turn_domain])
                    '''
                    example:

                    for dial_turn
                    {'text': 'i found 1 cheap hotel for you that includes parking . do you like me to book it ?', 
                    'metadata': {'taxi': {'book': {'booked': []}, 'semi': {'leaveat': '', 'destination': '', 'departure': '', 'arriveby': ''}}, 'police': {'book': {'booked': []}, 'semi': {}}, 'restaurant': {'book': {'booked': [], 'time': '', 'day': '', 'people': ''}, 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}}, 'hospital': {'book': {'booked': []}, 'semi': {'department': ''}}, 
                    'hotel': {'book': {'booked': [], 'stay': '', 'day': '', 'people': ''}, 'semi': {'name': 'not mentioned', 'area': 'not mentioned', 'parking': 'yes', 'pricerange': 'cheap', 'stars': 'not mentioned', 'internet': 'not mentioned', 'type': 'hotel'}}, 'attraction': {'book': {'booked': []}, 'semi': {'type': '', 'name': '', 'area': ''}}, 'train': {'book': {'booked': [], 'people': ''}, 'semi': {'leaveat': '', 'destination': '', 'day': '', 'arriveby': '', 'departure': ''}}}, 
                    'dialog_act': {'booking-inform': [['none', 'none']], 'hotel-inform': [['price', 'cheap'], ['choice', '1'], ['parking', 'none']]}, 
                    'span_info': [['hotel-inform', 'price', 'cheap', 3, 3], ['hotel-inform', 'choice', '1', 2, 2]]}

                    'user': 'no , i just need to make sure it is cheap . oh , and i need parking', 
                    'user_delex': 'no , i just need to make sure it is [value_pricerange] . oh , and i need parking', 
                    'resp': 'i found [value_choice] [value_price] hotel for you that include -s parking . do you like me to book it ?', 
                    'resp_nodelex': 'i found 1 cheap hotel for you that include -s parking . do you like me to book it ?', 
                    'pointer': '0,1,0,0,0,0', 
                    'match': '1', 
                    'constraint': '[hotel] pricerange cheap type hotel parking yes', 
                    'cons_delex': '[hotel] pricerange type parking', 
                    'sys_act': '[hotel] [inform] price choice parking [offerbook]', 
                    'turn_num': 1, 
                    'turn_domain': '[hotel]'
                    '''

                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict) #if not cfg.enable_multi_cntfact else [copy.deepcopy(constraint_dict) for _ in range(cfg.topk_cntfact)]
                    if cfg.cntfact_max_mode:
                        if cntfact_active:
                            prev_cntfact_constraint_dict = copy.deepcopy(cntfact_constraint_dict)
                        else:
                            prev_cntfact_constraint_dict = copy.deepcopy(constraint_dict)

                    if 'user' in single_turn:
                        dial['log'].append(single_turn)
                        for t in single_turn['user'].split() + single_turn['resp'].split() + constraints + sys_act:
                            self.vocab.add_word(t)
                        for t in single_turn['user_delex'].split():
                            if '[' in t and ']' in t and not t.startswith('[') and not t.endswith(']'):
                                single_turn['user_delex'].replace(t, t[t.index('['): t.index(']')+1])
                            elif not self.vocab.has_word(t):
                                self.vocab.add_word(t)

                    single_turn = {}


            data[fn] = dial
            # pprint(dial)
            # if count == 20:
            #    break
        if not cfg.cntfact_max_mode:
            self.vocab.construct() # may omit cntfact vocab not include in ori vocab set
            self.vocab.save_vocab('data/multi-woz-processed/vocab')
            with open('data/multi-woz-analysis/dialog_acts.json', 'w') as f:
                json.dump(ordered_sysact_dict, f, indent=2)
            with open('data/multi-woz-analysis/dialog_act_type.json', 'w') as f:
                json.dump(self.unique_da, f, indent=2)
        return data


if __name__=='__main__':
    db_paths = {
            'attraction': 'db/attraction_db.json',
            'hospital': 'db/hospital_db.json',
            'hotel': 'db/hotel_db.json',
            'police': 'db/police_db.json',
            'restaurant': 'db/restaurant_db.json',
            'taxi': 'db/taxi_db.json',
            'train': 'db/train_db.json',
        }
    #pdb.set_trace()
    get_db_values('db/value_set.json')
    if not cfg.skip_preprocess:
        preprocess_db(db_paths)
    dh = DataPreprocessor()
    data = dh.preprocess_main()
    if not os.path.exists('data/multi-woz-processed'):
        os.mkdir('data/multi-woz-processed')

    if cfg.cntfact_max_mode:
        with open(cfg.cntfact_max_save_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open('data/multi-woz-processed/data_for_damd.json', 'w') as f:
            json.dump(data, f, indent=2)

