from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from collections import defaultdict
import torch
import stanza
import re

class ROBertaUkr:
    def __init__(self):
        self.model_name = "nanigock/ukr-roberta-ner-finetuned"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        self.model = RobertaForTokenClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = self.model.config.id2label
        self.model.to(self.device)
        # stanza.download("uk")
        self.nlp = stanza.Pipeline("uk", processors="tokenize,mwt,pos,lemma")

    def mask_entities_in_sentence(self, sentence, max_len=512, lemmatize=True):
        if lemmatize:
            def normalize_entity(text):
                doc = self.nlp(text)
                return [word.lemma for sent in doc.sentences for word in sent.words][0]
        self.model.eval()
        
        words = sentence.strip().split()
        encoding = self.tokenizer(words,
                            is_split_into_words=True,
                            return_offsets_mapping=True,
                            padding="max_length",
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt")
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0]

        predicted_labels = []
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            label = self.id2label[predictions[idx].item()]
            predicted_labels.append(label)
            previous_word_idx = word_idx

        def anonymize_tokens(tokens, tags):
            entities = []
            current_entity = []
            current_label = None

            for token, tag in zip(tokens, tags):
                if tag.startswith("B-"):
                    if current_entity:
                        entities.append((" ".join(current_entity), current_label))
                        current_entity = []
                    current_entity = [token]
                    current_label = tag[2:].lower()
                elif tag.startswith("I-") and current_label == tag[2:].lower():
                    current_entity.append(token)
                else:
                    if current_entity:
                        entities.append((" ".join(current_entity), current_label))
                        current_entity = []
                        current_label = None

            if current_entity:
                entities.append((" ".join(current_entity), current_label))

            label_counters = defaultdict(int)
            entity_to_placeholder = {}

            for text, label in entities:
                if lemmatize:
                    text = normalize_entity(text.lower())
                if text not in entity_to_placeholder:
                    label_counters[label] += 1
                    entity_to_placeholder[text] = f"[{label}_{label_counters[label]}]"

            result = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                tag = tags[i]
                if tag.startswith("B-"):
                    entity = [token]
                    label = tag[2:].lower()
                    i += 1
                    while i < len(tokens) and tags[i].startswith("I-"):
                        entity.append(tokens[i])
                        i += 1
                    entity_text = " ".join(entity)
                    if lemmatize:
                        lemma_key = normalize_entity(entity_text.lower())
                        result.append(entity_to_placeholder[lemma_key])
                    else:
                        result.append(entity_to_placeholder[entity_text])
                else:
                    result.append(token)
                    i += 1

            return " ".join(result), entity_to_placeholder

        masked_sentence, entity_mapping = anonymize_tokens(words, predicted_labels)

        return masked_sentence, entity_mapping


    def process_large_text_in_blocks(self, text, max_len_block=512, max_len_roberta= 512, lemmatize=False):
        words = text.strip().split()
        block_start_idx = 0
        blocks = []
        current_block = []

        while block_start_idx < len(words):
            current_block = words[block_start_idx:(block_start_idx + max_len_block)]
            blocks.append(current_block)
            block_start_idx += len(current_block)
        combined_entity_mapping = {}
        masked_sentences = []

        for block in blocks:
            block_text = " ".join(block)
            masked_sentence, entity_mapping = self.mask_entities_in_sentence(block_text, max_len=max_len_roberta, lemmatize=lemmatize)
            masked_sentences.append(masked_sentence)
            
            for entity, placeholder in entity_mapping.items():
                if entity not in combined_entity_mapping:
                    combined_entity_mapping[entity] = placeholder

        full_masked_sentence = " ".join(masked_sentences)

        return full_masked_sentence, combined_entity_mapping

    def unify_entity_replacements(self, entity_mapping, lemmatize=True):
        
        if lemmatize:
            stanza.download("uk")
            nlp = stanza.Pipeline("uk", processors="tokenize,mwt,pos,lemma")
            
            def normalize_entity(text):
                doc = nlp(text)
                return [word.lemma for sent in doc.sentences for word in sent.words][0]
        
        unified_mapping = {}
        entity_to_placeholder = {}
        for entity, placeholder in entity_mapping.items():
            entity = self.remove_punctuation(entity)
            print(entity)
            normalized_entity = normalize_entity(entity) if lemmatize else entity
            cathegory = placeholder.split("_")[0][1:]
            if normalized_entity not in entity_to_placeholder:
                counter = len(entity_to_placeholder)
                entity_to_placeholder[normalized_entity] = f"{cathegory}_{counter}"            
            unified_mapping[entity] = f"[{entity_to_placeholder[normalized_entity]}]"
            unified_mapping[normalized_entity] = f"[{entity_to_placeholder[normalized_entity]}]"
        return unified_mapping

    def replace_entities_in_text(self, text, unified_mapping):
        def replace_entity(match):
            entity = match.group(0)
            return unified_mapping.get(entity, entity)
        pattern = re.compile(r'\b(' + '|'.join(re.escape(entity) for entity in unified_mapping.keys()) + r')\b')
        replaced_text = re.sub(pattern, replace_entity, text)
        return replaced_text
    
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)
    
    def run(self, sentence, max_len_block=100, lemmatize=False):
        masked_sentence, entity_map=self.process_large_text_in_blocks(sentence, max_len_block, lemmatize)
        unified_mapping = self.unify_entity_replacements(entity_map)
        masked_sentence = self.replace_entities_in_text(sentence, unified_mapping)
        return masked_sentence, unified_mapping
    

# sentence = """

# Цього року сім’я Петрових вирушає на відпочинок до Італії. Вони планують відпочити на морі, побувати в Римі та Флоренції. Тато, Ігор, працює інженером у великій компанії і зазвичай має багато роботи, але цього разу він вирішив взяти відпустку, щоб провести час з родиною.

# Мама, Олена, працює лікарем у місцевій клініці. Вона давно мріяла поїхати в Італію, щоб побачити Колізей та насолодитися італійською кухнею. Цей відпочинок стане чудовою нагодою для неї відпочити від важкої роботи.

# Їхній син, Андрій, навчається в університеті та займається волейболом. Він із задоволенням їде до Італії, бо чує багато хороших відгуків про пляжі та атмосферу цієї країни. Окрім цього, він збирається відвідати кілька університетів у Римі, щоб подивитися на місцеву студентську життя.

# Молодша донька, Марічка, ще ходить до школи, але дуже любить подорожувати. Вона мріє побачити Пізанську вежу і зробити багато фотографій на фоні історичних пам’яток. Всі члени сім’ї з нетерпінням чекають на цей відпочинок, і вони сподіваються, що цей рік принесе багато позитивних емоцій.

# """


# roberta_infer = ROBertaUkr()
# masked_text, unified_mapping = roberta_infer.run(sentence)
# print(masked_text)
# print(unified_mapping)




