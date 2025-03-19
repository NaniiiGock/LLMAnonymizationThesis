
from datetime import datetime
from dotenv import load_dotenv
from processing.pattern_processing import PatternProcessor
from processing.ner_processing import NERProcessor
from processing.postprocessor import PostProcessor
from providers.openai_provider import OpenAIProvider
from processing.retriever import Retriever
import yaml
import json

def load_pipeline_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)
    
load_dotenv()

class PrivacyPipeline:
    def __init__(self, config_path):
        self.config = load_pipeline_config(config_path)

        self.entity_map = {}
        self.reverse_map = {}

        self.pattern_processor = PatternProcessor(self.config['pattern_processor'])
        self.ner_processor = NERProcessor(self.config['ner_processor'])
        self.post_processor = PostProcessor(self.config['postprocessor']['mode'])
        self.openai = OpenAIProvider(self.config['llm_invoke'])
        self.retriever = Retriever(self.config["retriever"])
        self.logging_enabled = self.config['logging']['enabled']
        self.log_path = self.config['files']['log_path']
    
    def preprocess_input(self, text):

        pattern_replacements = {}
        processed_text = text
        
        processed_text, pattern_replacements, entity_map = self.pattern_processor.preprocess(processed_text)
        self.entity_map = entity_map

        processed_text, ner_replacements, entity_map = self.ner_processor.preprocess(processed_text, pattern_replacements, entity_map)

        self.reverse_map = {v: k for k, v in self.entity_map.items()}
        self.replacements = {**pattern_replacements, **ner_replacements}
        
        return processed_text, {**pattern_replacements, **ner_replacements}
    
        
    def preprocess_task(self, task, replacements):
        processed_task = task 
        for original, replacement in replacements.items():
            if original in task:
                processed_task = task.replace(original, replacement)
        return processed_task
    
    def prepare_prompt(self, anonymized_text: str, task_description: str) -> str:
        privacy_instruction = """
        Process this text while maintaining privacy. Do not attempt to:
        1. Reverse any anonymized tokens
        2. Generate or infer personal information
        3. Include specific details about anonymized entities
        """
        return f"{privacy_instruction}\n\nTask: {task_description}\n\nText: {anonymized_text}"
    
        # privacy_instruction = """
        # Process this text while maintaining privacy. Do not attempt to:
        # 1. Reverse any anonymized tokens
        # 2. Generate or infer personal information
        # 3. Include specific details about anonymized entities
        # """
        # return f"Завдання: {task_description}\n\nТекст: {anonymized_text}"

    
    async def invoke(self, prompt):
        return await self.openai.query_llm(prompt)
    
    def postprocess_output(self, llm_output, context):
        return self.post_processor.postprocess_output(llm_output, context)

    def log_interaction(self, results):
        if not self.logging_enabled:
            return
        
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'processing_status': 'success' if 'error' not in results else 'error',
            'steps_completed': [step['step'] for step in results['processing_steps']],
        }
        
        if self.config['logging'].get('include_entity_counts', True):
            entity_types = {}
            for token in self.entity_map:
                entity_type = token.split('_')[0][1:]
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
            log_entry['entity_counts'] = entity_types
            
        with open(self.log_path, 'a') as log_file:
            log_file.write(json.dumps(log_entry) + '\n')


    async def process_pipeline(self, user_input, task, uploaded_files):
        results = {
            "original_input": user_input,
            "original_task": task,
            "processing_steps": []
        }
        
        current_input = user_input
        current_task = task
        current_replacements = {}
        current_entity_map = {}

        processing_order = self.config["processing"]["order"]

        for step in processing_order:
            if step == "pattern_processor":
                current_input, replacements, entity_map = \
                    self.pattern_processor.preprocess(current_input)
                
                current_replacements.update(replacements)
                current_entity_map.update(entity_map)

                print("====================")
                print("PATTERN PROCESSOR")
                print(current_replacements)
                print(current_entity_map)
                print("====================")

                results["processing_steps"].append({
                        "step": "pattern_processor",
                        "entities_found": len(replacements)
                    })
                
            elif step == "ner_processor":
                current_input, ner_replacements, entity_map = \
                    self.ner_processor.preprocess(current_input, current_replacements, current_entity_map )
                
                current_replacements.update(ner_replacements)
                current_entity_map.update(entity_map)

                print("====================")
                print("NER PROCESSOR")
                print(current_replacements)
                print(current_entity_map)
                print("====================")

                results["processing_steps"].append({
                    "step": "ner_processor",
                    "entities_found": len(ner_replacements)
                })

            elif step == "llm_invoke":
                print("====================")
                print("LLM INVOKE")
                print(current_replacements)
                current_task = self.preprocess_task(current_task, current_replacements)
                print(current_task)

                prompt = self.prepare_prompt(current_input, current_task)
                print(prompt)

                llm_response = await self.invoke(prompt)
                print(llm_response)
                print("====================")

                results["processing_steps"].append({
                        "step": "llm_processor",
                        "success": llm_response is not None
                    })

            elif step == "postprocessor":
                final_output = self.postprocess_output(llm_response, current_replacements)
                print("====================")
                print("FINAL OUTPUT")
                print(final_output)
                print("====================")
                results["processing_steps"].append({
                    "step": "postprocessor"
                })

            elif step == "retrieve":
                retrieved_texts = self.retriever.run_retriever(user_input, task, "/Users/lilianahotsko/Desktop/University/UCU/LLMAnonymizationUV/uploaded_files")
                print("====================")
                print("RETRIEVED TEXTS")
                print(retrieved_texts)
                print("====================")
                current_input += f"BASE KNOWLEDGE: {retrieved_texts}"
                results["processing_steps"].append({
                    "step": "retriever",
                    "retrieved_texts" : retrieved_texts
                })
                
        results["anonymized_input"] = current_input
        results["anonymized_task"] = current_task
        results["llm_response"] = llm_response
        results["final_output"] = final_output
        
        if self.logging_enabled:
            self.log_interaction(results)
            
        return results