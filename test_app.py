from privacy_pipeline import PrivacyPipeline
import os

async def main():
    config_path = "temp_config.yml"
    pipeline = PrivacyPipeline(config_path)
    
    input_path = pipeline.config['files']['input_path']
    task_path = pipeline.config['files']['task_path']
    output_path = pipeline.config['files']['output_path']
    
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r') as f:
        user_input = f.read()

    with open(task_path, 'r') as f:
            task = f.read()

    results = await pipeline.process_pipeline(user_input, task)
    
    with open(output_path, 'w') as f:
        f.write(results.get('final_output', 'Error: No output generated'))
    
    print("\nPipeline Results:")
    print(f"Original Input: \n{results['original_input']}")
    print(f"Anonymized Input: \n{results['anonymized_input']}")
    print(f"LLM response: \n {results['llm_response']}")
    print(f"Final Output: \n{results.get('final_output', 'Error')}")
    print(f"Output written to: \n{output_path}")

if __name__ == "__main__":
    import asyncio
    results = asyncio.run(main())