import re
from typing import Dict, Tuple

class PIIProtector:
    """
    Utility to anonymize and de-anonymize sensitive Indian PII 
    (Phone numbers, Email addresses, Aadhaar, PAN, and names/prefixes).
    """
    def __init__(self):
        # Regexes for Indian PII
        self.patterns = {
            "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "PHONE": r'\b(?:\+?91[\-\s]?)?[6789]\d{9}\b',
            "AADHAAR": r'\b\d{4}[\-\s]\d{4}[\-\s]\d{4}\b|\b\d{12}\b',
            "PAN": r'\b[A-Z]{5}\d{4}[A-Z]\b',
            # Match common salutations followed by capitalized words (e.g. Mr. Rajesh Kumar)
            "SALUTATION_NAME": r'\b(?:Mr\.|Mrs\.|Ms\.|Shri|Smt\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            # Match parentage markers common in Indian law: S/o, D/o, W/o Rajesh Kumar
            "PARENTAGE_NAME": r'\b(?:[SDW]/o)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        }

    def anonymize(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Anonymizes PII in text and returns the anonymized text and a mapping dict
        to restore the original values later.
        """
        if not text:
            return text, {}

        mapping = {}
        anonymized_text = text
        token_counter = {}

        for pii_type, regex_str in self.patterns.items():
            token_counter[pii_type] = 0
            matches = list(re.finditer(regex_str, anonymized_text))
            
            # Sort matches in reverse order so replacements don't shift index positions of subsequent matches
            matches.sort(key=lambda x: x.start(), reverse=True)
            
            for match in matches:
                original_value = match.group(0)
                # Skip if already a token
                if original_value.startswith("[") and original_value.endswith("]"):
                    continue
                
                # Check if we already registered this value to use the same token
                existing_token = None
                for tok, val in mapping.items():
                    if val == original_value:
                        existing_token = tok
                        break
                
                if existing_token:
                    token = existing_token
                else:
                    token = f"[{pii_type}_{token_counter[pii_type]}]"
                    token_counter[pii_type] += 1
                    mapping[token] = original_value
                
                start, end = match.span()
                anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]

        return anonymized_text, mapping

    def deanonymize(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Restores the original PII values using the mapping dictionary.
        """
        if not text or not mapping:
            return text
            
        deanonymized_text = text
        for token, original_value in mapping.items():
            deanonymized_text = deanonymized_text.replace(token, original_value)
        return deanonymized_text

# Simple test execution if run directly
if __name__ == "__main__":
    protector = PIIProtector()
    sample = "Mr. Rajesh Kumar (PAN: ABCDE1234F, Aadhaar: 1234 5678 9012) resides with his father S/o Shri Suresh Kumar. Contact: +91 9876543210 or email rajesh@gmail.com."
    anon, mapping = protector.anonymize(sample)
    print("Anonymized:\n", anon)
    print("\nMapping:\n", mapping)
    print("\nDe-anonymized:\n", protector.deanonymize(anon, mapping))
