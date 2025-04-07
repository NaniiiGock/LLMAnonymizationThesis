prompt = """You are an anonymization assistant.

Your task is to identify and replace all entities in the following text that belong to any of the following categories, using a consistent identifier format. You must detect and anonymize:

- Persons (PER)
- Locations (LOC) — cities, countries, landmarks, addresses
- Organizations (ORG) — companies, institutions, government bodies
- Nationalities or religious/political groups (NORP)
- Dates (DATE)
- Events (EVENT)
- Products (PRODUCT)
- Laws (LAW)
- Languages (LANGUAGE)
- Works of art (WORK_OF_ART)
- Time values (TIME)
- Percentages (PERCENT)
- Money values (MONEY)
- Quantities (QUANTITY)

Each entity must be replaced with a unique identifier in the format:
- PER_1, PER_2, ...
- LOC_1, LOC_2, ...
- ORG_1, ORG_2, ...
- DATE_1, EVENT_1, etc.

Instructions:
- Replace every full and partial mention of the entity with the correct identifier.
  E.g., "Barack Obama" and "Obama" → PER_1
- Reuse the same identifier consistently across the entire text.
- Do **not** include code or explanations.
- Return **only**:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
     {"<original mention>", "<PII-style ID>"}

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"[PER_1] was born in [LOC_1] in [DATE_1]. [PER_1] was elected president in [DATE_2]."

JSON:
{
  "Barack Obama": "PER_1",
  "Obama": "PER_1",
  "Honolulu": "LOC_1",
  "1961": "DATE_1",
  "2008": "DATE_2"
}


Now process the following text:

"""

prompt_level_1_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **realistic but random** replacement of the same type. The replacement does **not** need to match the original context exactly, but must sound plausible and belong to the same category.

Category: {category}
Original: {entity}

Output:
Return only the replacement entity, no explanations.
"""

prompt_level_2_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **contextually appropriate** replacement that fits naturally into the surrounding text.

Category: {category}
Context: "{context}"

The replacement should sound natural in this context and match the type of entity ({category}).

Output:
Return only the replacement entity, no explanations.
"""

prompt_level_1_template = """
You are an assistant that replaces placeholders in text with similar values.
Replace with the random word that applies to the same detected cathegory.
Respond with the fully replaced version of the original text (do not include the placeholder version).

Instructions:
- Replace every full and partial mention of the entity with the updated identifier
- Reuse the same identifier consistently across the entire text.
- Do **not** include code or explanations.
- Do not ommit the entities, you must replace all of them.
- Return **only**:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
     {"entity": "<original mention>", "updated_entity": "<updated entity>"}

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"Joseph Biden was born in Scranton in 1942. Joseph was elected president in 2021."

JSON:
[
  {"entity": "Barack Obama", "updated_entity": "Joseph Biden "},
  {"entity": "Obama", "updated_entity": "Joseph"},
  {"entity": "Honolulu", "updated_entity": "Scranton"},
  {"entity": "1961", "updated_entity": "1942"},
  {"entity": "2008", "updated_entity": "2021"}
]

Now process the following text:

"""
