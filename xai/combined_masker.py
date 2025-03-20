
class CombinedMasker:
    def __init__(self, masker_text, masker_structured):
        self.masker_text = masker_text
        self.masker_structured = masker_structured

    def __call__(self, inputs):
        # Separate the inputs into text and structured parts
        text_input = inputs['input_ids'], inputs['attention_mask']
        structured_input = inputs['structured']
        
        # Apply the text masker and structured masker independently
        masked_text = self.masker_text(text_input)
        masked_structured = self.masker_structured(structured_input)
        
        # Combine the masked outputs
        return masked_text, masked_structured