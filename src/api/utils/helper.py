import re

def remove_special_characters(input_string):
    pattern = r"[\*\-\+\[\]\\\\]"
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string