
import spacy

class NERProcessor:
    def __init__(self, config):
        self.model_name = config['model']
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, processed_text, pattern_replacements, entity_map):
        doc = self.nlp(processed_text)
        ner_replacements = {}
        
        for ent in doc.ents:
            if ent.text not in pattern_replacements.values() and ent.text not in entity_map.values(): 
                replacement = f"[{ent.label_}_{len(entity_map)}]"
                processed_text = processed_text.replace(ent.text, replacement)
                entity_map[replacement] = ent.text
                ner_replacements[ent.text] = replacement


        return processed_text, ner_replacements, entity_map
    
