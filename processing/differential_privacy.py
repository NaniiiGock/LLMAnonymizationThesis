from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine, OperatorConfig
from pydp.algorithms.laplacian import BoundedMean
import numpy as np

# Initialize PII analyzer and anonymizer
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
deanonymizer = DeanonymizeEngine()

# Sample text containing PII
text = "John Doe's email is john.doe@example.com and his phone number is 555-1234."

# Step 1: PII Detection
analyzer_results = analyzer.analyze(text=text, language='en')

# Step 2: Data Anonymization
anonymized_results = anonymizer.anonymize(
    text=text,
    analyzer_results=analyzer_results,
    operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"})}
)
anonymized_text = anonymized_results.text
print("Anonymized Text:", anonymized_text)

# Step 3: Differential Privacy Application
# Example: Computing a differentially private mean of a dataset
# Here, we simulate a dataset of numerical values
data = np.array([10, 20, 30, 40, 50])
dp_mean = BoundedMean(epsilon=1.0, lower_bound=0, upper_bound=100)
dp_result = dp_mean.quick_result(data)
print("Differentially Private Mean:", dp_result)

# Step 4: Re-identification (Optional)
# Assuming we have a mapping of anonymized tokens to original PII
# For demonstration, we'll use a simple dictionary
pii_mapping = {"<ANONYMIZED>": "john.doe@example.com"}
deanonymized_text = deanonymizer.deanonymize(
    text=anonymized_text,
    entities=analyzer_results,
    pii_mapping=pii_mapping
)
print("Deanonymized Text:", deanonymized_text)
