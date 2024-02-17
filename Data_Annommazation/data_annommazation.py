from transformers import pipeline
import os
import sys

from Data_Annommazation.logger import logging
from Data_Annommazation.Exception import DataAnnommazationException


try:
    logging.info("Loading the model for Named Entity Recognition (NER)")
    # Load the model for Named Entity Recognition (NER)
    ner_model = pipeline("ner", grouped_entities=True)

except Exception as e:
    raise DataAnnommazationException(e, sys)

def anonymize_text(text):
    try:
        logging.info("Anonymizing the text")
    
        # Perform Named Entity Recognition (NER) to identify entities
        entities = ner_model(text)

        anonymized_text = text
        
        logging.info("Replacing identified entities with asterisks")
        
        # Replace identified entities with asterisks
        for entity in entities:
            start_idx = entity['start']
            end_idx = entity['end']
            entity_type = entity['entity_group']
            entity_text = entity['word']
            
            logging.info(f"Replacing {entity_text} with asterisks")

            # Anonymize if entity is a person, email, or phone number
            if entity_type in ['PER', 'EMAIL', 'PHONE']:
                anonymized_text = anonymized_text[:start_idx] + '*' * (end_idx - start_idx) + anonymized_text[end_idx:]

        logging.info("Anonymized text: " + anonymized_text)
        return anonymized_text
    
    except Exception as e:
        raise DataAnnommazationException(e, sys)