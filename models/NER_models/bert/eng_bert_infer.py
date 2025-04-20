import torch
from collections import defaultdict
from transformers import BertTokenizer, BertForTokenClassification
import re 

class BertEng:
    def __init__(self):
        self.model_name = "nanigock/bert-token-classifier-ner-v1"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForTokenClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = self.model.config.id2label
        self.model.to(self.device)

    def correct_mappings(self, sentence, current_mappings):
        def find_original_word(replacement, sentence):
            word = replacement.strip('[]')

            pattern_with_space = r'\b' + re.escape(word[:2]) + r'\s?' + re.escape(word[2:]) + r'\b'  # e.g., '13 September'
            pattern_without_space = r'\b' + re.escape(word) + r'\b'  # e.g., '13september'

            matches = re.finditer(pattern_with_space, sentence, re.IGNORECASE)
            for match in matches:
                return match.group(0)

            matches = re.finditer(pattern_without_space, sentence, re.IGNORECASE)
            for match in matches:
                return match.group(0)
            
            return None

        new_mappings = {}
        
        for original, replacement in current_mappings.items():
            original_word = find_original_word(original, sentence)
            if original_word:
                new_mappings[original_word] = replacement
        
        return new_mappings

    def mask_entities_in_sentence(self, sentence, max_len=512):
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        ids = inputs["input_ids"].to(self.device)
        mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(ids, mask)
        logits = outputs[0]
        active_logits = logits.view(-1, self.model.num_labels)
        predictions = torch.argmax(active_logits, axis=1)

        tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        labels = [self.id2label[i] for i in predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, labels))

        words, tags = [], []
        current_word = ""
        current_tag = None

        for tok, tag in wp_preds:
            if tok in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if tok.startswith("##"):
                current_word += tok[2:]
            else:
                if current_word:
                    words.append(current_word)
                    tags.append(current_tag)
                current_word = tok
                current_tag = tag
        if current_word:
            words.append(current_word)
            tags.append(current_tag)

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
                normalized_text = text.replace(" ", "").lower()
                
                if normalized_text not in entity_to_placeholder:
                    label_counters[label] += 1
                    entity_to_placeholder[normalized_text] = f"[{label}_{label_counters[label]}]"

            result = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                tag = tags[i]                
                token_normalized = token.replace("##", "").lower()
                
                if tag.startswith("B-"):
                    entity = [token]
                    label = tag[2:].lower()
                    i += 1
                    while i < len(tokens) and tags[i].startswith("I-"):
                        entity.append(tokens[i])
                        i += 1
                    entity_text = " ".join(entity)
                    entity_text_normalized = entity_text.replace(" ", "").lower()
                    result.append(entity_to_placeholder.get(entity_text_normalized, f"[{label}_unknown]"))
                else:
                    result.append(token)
                    i += 1

            return " ".join(result), entity_to_placeholder
            
        masked_sentence, entity_mapping = anonymize_tokens(words, tags)
        corrected_entity_mapping = self.correct_mappings(sentence, entity_mapping)

        return masked_sentence, corrected_entity_mapping


# sentence = """
# 1914 all over again. On 13 September, Chamberlain informed the German government that he was willing to go to Germany to discuss the crisis personally with Hitler. Two days later, the sixty-nine-year-old Prime Mini ster, in his bid to secure European peace, made his very first flight—from Croydon to Munich—and was then transported in Hitler’s special train to Berchtesgaden. In the ensuing talks, the two leaders came to an', 'agreement that any districts in Czechslovakia with a German  majority which opted for self-determination should be peacefully transf erred to the German Reich, and, on his return to England, Chamberlain spent the next week putting pressure on the French and Czech governments to agree to this proposal. When he returned to Germany on 22 September, however, to report his success to Hitler, he was greatly taken aback when a', 'May 1937  Chamberlain succeeds Baldwin as Prime Minister in Britain July 1937  Japanese troops invade Chinese mainland from Manchuria October 1937  Italy adheres to anti-comintern pact December 1937  Italy announces her intention of withdrawing from the League February 1938  Chamberlain and Mussolini agree to recognition of Italian conquest of Abyssinia and to withdrawal of 10,000 Italian troops from Spain. This agreement leads to resignation of British Foreign Secretary Eden', 'one: he believes that it would have been possible for Chamberlain after March 1939 to secure sufficient support in parliament and in the country at large for a close alliance with France and for a policy of containing and encircling Germany within the framework of The origins of the second world war 1933–1941     60robin-bobin', 'Chamberlain contacted Mussolini to enlist his support in a bid to persuade Hitler to resume negotiations rather than to resort to force. The Prime Minister’s emissary, Sir Horace Wilson, duly passed on to Hitler in a tense meeting on 27 September Chamberlain’s message that ‘If in pursuit of her Treaty obligations, France became actively engaged in hostilities against German y, the United Kingdom would feel obliged.
# """
# bert_eng = BertEng()
# masked_sentence, mapping = bert_eng.mask_entities_in_sentence(sentence,max_len=512)

# print(masked_sentence)
# print(mapping)