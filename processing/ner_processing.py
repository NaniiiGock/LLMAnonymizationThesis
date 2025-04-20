
import spacy
from models.NER_models.bert.eng_bert_infer import BertEng
from models.NER_models.roberta_ukr.ukr_roberta import ROBertaUkr

class NERProcessor:
    def __init__(self, config=None):
        self.model_name = config.get('model', "")

    def preprocess(self, processed_text, pattern_replacements, entity_map):
        if self.model_name in ["en_core_web_sm", "uk_core_news_sm"]:
            nlp = spacy.load(self.model_name)
            doc = nlp(processed_text)
            ner_replacements = {}
            
            for ent in doc.ents:
                if ent.text not in pattern_replacements.values() and ent.text not in entity_map.values(): 
                    replacement = f"[{ent.label_}_{len(entity_map)}]"
                    processed_text = processed_text.replace(ent.text, replacement)
                    entity_map[replacement] = ent.text
                    ner_replacements[ent.text] = replacement
            return processed_text, ner_replacements, entity_map
        
        elif self.model_name == "bert_eng":
            bert_infer = BertEng()
            masked_sentence, ner_replacements = bert_infer.mask_entities_in_sentence(processed_text, max_len=128)
            final_replacements = {}
            for entity, cathegory in ner_replacements.items():
                if entity not in pattern_replacements.values() and entity not in entity_map.values():
                    cat_name = cathegory[1:-1].split("_")[0].upper()
                    replacement = f"{cat_name}_{len(entity_map)}"
                    masked_sentence = masked_sentence.replace(cathegory, f"[{replacement}]")
                    entity_map[replacement] = entity
                    final_replacements[entity] = replacement
            return masked_sentence, final_replacements, entity_map

        elif self.model_name == "roberta_ukr":
            roberta_infer = ROBertaUkr()
            masked_sentence, ner_replacements = roberta_infer.run(processed_text)
            for entity, cathegory in ner_replacements.items():
                if entity not in pattern_replacements.values() and entity_map.values():
                    replacement = cathegory.split("_")[0].upper() + "_" + len(entity_map)
                    entity_map[replacement] = entity
                    ner_replacements[entity] = replacement
            return masked_sentence, ner_replacements, entity_map
        
    def replace_entities_with_masks(self, text, mapping):
        for entity, mask in mapping.items():
            text = text.replace(entity, mask)
        return text
        