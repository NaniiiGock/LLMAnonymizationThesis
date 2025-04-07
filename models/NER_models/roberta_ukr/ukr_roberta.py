from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from collections import defaultdict
import torch

class ROBertaUkr:
    def __init__(self):
        self.model_name = "nanigock/ukr-roberta-ner-finetuned"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
        self.model = RobertaForTokenClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2label = self.model.config.id2label
        self.model.to(self.device)

    def mask_entities_in_sentence(self, sentence, max_len=128, lemmatize=True):
        
        if lemmatize:
            import stanza
            stanza.download("uk")
            nlp = stanza.Pipeline("uk", processors="tokenize,mwt,pos,lemma")

            def normalize_entity(text):
                doc = nlp(text)
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

sentence = "Дідам співав Матвій, а Марічка з Устимом їхали до Житомира. З Матвієм сиділи діди. В Житомирі сьогодні світить сонце."
roberta_infer = ROBertaUkr()
masked_sentence, entity_map = roberta_infer.mask_entities_in_sentence(sentence, max_len=128, lemmatize=False)
print(masked_sentence)
print(entity_map)
