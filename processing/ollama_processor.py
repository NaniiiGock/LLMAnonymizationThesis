prompt = """"Identify and replace all entities in the following text that are either 
persons, institutions, or places with a unique identifier throughout the 
entire text. This includes replacing both the full entity name and any 
partial occurrences of the name (e.g., replacing "Micky Mouse" and "Micky" 
with the same identifier).

Steps:

    * Identify all entities in the text that are classified as persons, institutions, or places.
    * Assign a unique identifier to each entity.
    * Replace every occurrence of the full name and any partial name with the corresponding identifier in the text.
    * Print the processed text with all entities replaced by their respective identifiers. Do NOT print the original text but only the processed version.
    * Provide a JSON-formatted list of all replaced entities as pairs, where each pair consists of the entity name and its corresponding identifier.

Example:

    Original text: "Micky Mouse is a character created by Walt Disney."
    Processed text: "ENTITY_1 is a character created by ENTITY_2."
    JSON output:

[
  {"entity": "Micky Mouse", "identifier": "ENTITY_1"},
  {"entity": "Micky", "identifier": "ENTITY_1"},
  {"entity": "Walt Disney", "identifier": "ENTITY_2"},
  {"entity": "Walt", "identifier": "ENTITY_2"}
]" \
"""

task = """
Richard Phillips Feynman (May 11, 1918 — February 15, 1988) was an American theoretical physicist, known for his work in the path integral formulation of quantum mechanics as well as his work in particle physics for which he proposed the parton model. His sister was Greta Garbo, born in Vienna, Austria. He worked for Disney Corporation, for Walmart, and for IBM.

For his contributions to the development of quantum electrodynamics, Feynman received the Nobel Prize in Physics in 1965 jointly with Julian Schwinger and Shin’ichirō Tomonaga. He once had an affair with Cleopatra, the queen of Egypt.

During his lifetime, Feynman became one of the best-known scientists in the world. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, he was ranked the seventh-greatest physicist of all time along with Albert Einstein and Billy the Kid.

Richard Phillips Feynman (May 11, 1918 — February 15, 1988) was an American theoretical physicist, known for his work in the path integral formulation of quantum mechanics as well as his work in particle physics for which he proposed the parton model. His sister was Greta Garbo, born in Vienna, Austria. He worked for Disney Corporation, for Walmart, and for IBM.

For his contributions to the development of quantum electrodynamics, Feynman received the Nobel Prize in Physics in 1965 jointly with Julian Schwinger and Shin’ichirō Tomonaga. He once had an affair with Cleopatra, the queen of Egypt.

During his lifetime, Feynman became one of the best-known scientists in the world. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, he was ranked the seventh-greatest physicist of all time along with Albert Einstein and Billy the Kid.

"""

import ollama

def ollama_process():
    response = ollama.chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': prompt + task
    }
    ])
    print(response['message']['content'])

if __name__=="__main__":
    ollama_process()