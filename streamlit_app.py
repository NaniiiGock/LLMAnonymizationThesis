import streamlit as st
import yaml
from streamlit_sortables import sort_items
import asyncio
from privacy_pipeline import PrivacyPipeline
import os
# =================== CONFIG LOAD ===================
CONFIG_FILE = "running_config_eng.yml"

def load_config():
    try:
        with open(CONFIG_FILE, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error reading config file: {e}")
        return {}
    
def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
    st.session_state.config = yaml.safe_load(CONFIG_FILE)

config = load_config()


if 'config' not in st.session_state:
    with open('running_config_eng.yml', 'r') as file:
        st.session_state.config = yaml.safe_load(file)

st.title("Configuration Editor")
st.sidebar.header("Edit Configuration")

# =================== SIDEBAR ===================

with st.sidebar.expander("Pattern Processor", expanded=False):
    custom_patterns = config.get("pattern_processor", {}).get("custom_patterns", {})

    new_key = st.text_input("New Pattern Key", "")
    new_value = st.text_input("New Pattern Value (Regex)", "")
    if st.button("Add Pattern") and new_key and new_value:
        custom_patterns[new_key] = new_value

    for key, value in list(custom_patterns.items()):
        custom_patterns[key] = st.text_input(f"{key}", value)
    config["pattern_processor"] = {"custom_patterns": custom_patterns}

with st.sidebar.expander("NER Processor", expanded=False):
    model = st.selectbox("NER Model", ["en_core_web_sm", "uk_core_news_sm"],
                         index=["en_core_web_sm", "uk_core_news_sm"].index(config.get("ner_processor", {}).get("model", "en_core_web_sm")))
    config["ner_processor"] = {"model": model}

    entity_types = config.get("ner_processor", {}).get("entity_types", [])
    new_entity = st.text_input("New Entity Type", "")
    if st.button("Add Entity") and new_entity:
        entity_types.append(new_entity)
    entity_types = st.multiselect("Entity Types", ["PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY"] + entity_types, entity_types)
    config["ner_processor"]["entity_types"] = entity_types

    sensitivity_levels = config.get("ner_processor", {}).get("sensitivity_levels", {})
    for entity in entity_types:
        sensitivity_levels[entity] = st.slider(f"{entity} Sensitivity", 1, 5,
                                               sensitivity_levels.get(entity, 3))
    config["ner_processor"]["sensitivity_levels"] = sensitivity_levels

with st.sidebar.expander("LLM Invoke", expanded=False):
    llm_provider = st.selectbox("Provider", ["openai", "anthropic", "local"],
                                index=["openai", "anthropic", "local"].index(config.get("llm_invoke", {}).get("provider", "openai")))
    llm_model = st.text_input("LLM Model", config.get("llm_invoke", {}).get("model", "gpt-4"))
    temperature = st.slider("Temperature", 0.0, 1.0, config.get("llm_invoke", {}).get("temperature", 0.3))
    max_tokens = st.number_input("Max Tokens", 100, 2000, config.get("llm_invoke", {}).get("max_tokens", 1000))

    config["llm_invoke"] = {
        "provider": llm_provider,
        "model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

with st.sidebar.expander("Logging", expanded=False):
    logging_enabled = st.checkbox("Enable Logging", config.get("logging", {}).get("enabled", True))
    logging_level = st.selectbox("Logging Level", ["DEBUG", "INFO", "WARNING", "ERROR"],
                                 index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("logging", {}).get("level", "INFO")))
    config["logging"] = {"enabled": logging_enabled, "level": logging_level}

if st.sidebar.button("Save Config"):
    save_config(config)
    st.sidebar.success("Configuration saved successfully!")


# =================== MAIN PAGE ===================

order = config.get("processing", {}).get("order", [])
order = sort_items(order)
config["processing"] = {"order": order}

st.title("Chat with Backend Processing")

uploaded_files = st.file_uploader("Upload Files - only while RETRIEVER step is used", type=None, accept_multiple_files=True)
upload_folder = "uploaded_files"
os.makedirs(upload_folder, exist_ok=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Saved file: {uploaded_file.name}")

user_input = st.text_area("User Input", height=150)
task_description = st.text_area("Task Description", height=100)

if st.button("Submit"):
    if user_input.strip() == "" or task_description.strip() == "":
        st.error("Both User Input and Task Description are required.")

    else:

        pipeline = PrivacyPipeline(CONFIG_FILE)

        async def process_input():
            results = await pipeline.process_pipeline(user_input, task_description, upload_folder)
            return results

        results = asyncio.run(process_input())

        st.subheader("Anonymized Input")
        st.write(results.get('anonymized_input', 'No anonymized input available.'))

        st.subheader("LLM Response")
        st.write(results.get('llm_response', 'No LLM response available.'))

        st.subheader("Final Output")
        st.write(results.get('final_output', 'No final output available.'))

st.subheader("Updated Configuration")
st.code(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))
save_config(config)
