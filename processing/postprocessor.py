
class PostProcessor:
    def __init__(self, mode):
        pass
    
    def postprocess_output(self, llm_output, context):
        processed_output = llm_output
        print("RECEIVED REPLACEMENTS: ", context)
        for original, replacement in context.items():
            if replacement in processed_output:
                processed_output = processed_output.replace(replacement, original)
        
        return processed_output