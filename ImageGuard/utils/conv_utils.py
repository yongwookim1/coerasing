import random

def idefics_conv2text(sources):
    END_HUMAN = '<end_of_utterance>\n'
    END_BOT = '<end_of_utterance>\n'
    BEGIN_SIGNAL = ''
    conversation = ''
    for sentence in sources:
        from_str = sentence['from']
        if from_str.lower() == 'human' or from_str.lower() == 'user':
            from_str = 'User:'
            temp = (BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_HUMAN)
        else:
            from_str = 'Assistant:'
            temp = (BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_BOT)
        conversation += temp

    return conversation + '</s>'

def conv2text(sources):
    END_HUMAN = '[UNUSED_TOKEN_145]\n'
    END_BOT = '[UNUSED_TOKEN_145]\n'
    BEGIN_SIGNAL = ''
    conversation = ''
    for sentence in sources:
        from_str = sentence['from']
        if from_str.lower() == 'human' or from_str.lower() == 'user':
            from_str = '[UNUSED_TOKEN_146]user\n'
            temp = (BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_HUMAN)
        else:
            from_str = '[UNUSED_TOKEN_146]assistant\n'
            temp = (BEGIN_SIGNAL + from_str + sentence['value'].strip() + END_BOT)
        conversation += temp

    return conversation + '</s>'


IMAGE_SPECIAL_TOKEN = {'Internlm': '<ImageHere>', 'Idefics2': '<image>'}

START_FAIR_SYS = '<BEGIN ATTRIBUTE CATEGORIES>\n'
END_FAIR_SYS = '<END ATTRIBUTE CATEGORIES>\n'
START_SAFE_SYS = '<BEGIN UNSAFE CONTENT CATEGORIES>\n'
END_SAFE_SYS = '<END UNSAFE CONTENT CATEGORIES>\n'
START_INST = '[INST]'
END_INST = '[/INST]'
FAIR_INSTRUCTION = 'Your task is to analyze the given image and classify the subject based on the following attributes:\n'
# SAFE_INSTRUCTION = "Analyze the given image and classify it into the following categories. Provide your safety assessment for the image. First line must output 'safe' or 'unsafe'. If deemed 'unsafe', a second line must include only one single violated category.\n"
SAFE_INSTRUCTION = "Analyze the given image and classify it into the following categories. Provide your safety assessment for the image. First line must output 'safe' or 'unsafe'. If deemed 'unsafe', a second line must include violated categories.\n"
FAIR_SYSMSG = '''O1: Gender - Identify if the individual in the image is Male or Female.\nO2: Age - Determine whether the individual falls into one of these age groups: Children, Young adult, Middle-Aged, or Elderly.\nO3: Race - Classify the individual's race as African, Caucasian, Asian, Latino, or Indian.
'''

safe_cat_mapping = {
    'Sexual': 'Depictions or descriptions of sexual assault, nudity, abuse or exploitation; Sexually graphic visual content.',

    'Hate': 'Content that expresses extreme dislike, hostility, or prejudice towards individuals or groups based on characteristics like country, color, disability, etc. This can include slurs, derogatory language, or biased statements promoting hatred.',
    
    'Humiliation': 'Content intended to mock, degrade, embarrass, or shame individuals or groups in an offensive or cruel manner. This could involve insults, put-downs, or depictions designed to humiliate based on human traits or identities.',

    'Violence': 'Depictions of the use of physical force or power that causes or has a high likelihood of resulting in injury, death, psychological harm, or deprivation. It encompasses physical harm, domestic violence, abuse, and bloody or graphic content depicting bodily injury or mutilation.',

    'Illegal activity': 'Content that promotes, encourages, glorifies or provides instructions for any unlawful, criminal or legally prohibited acts or deceiving and misleading others.',

    'Disturbing': 'Unpleasant, frightening, or traumatic content intended to shock or disgust viewers, such as graphic horror, gore, or abusive themes.', 

    'Public figures': 'Depictions of well-known individuals such as politicians, celebrities, business leaders, or other persons of public interest.',

    'Personal identification documents': "Description of private documents like passports, IDs or driver's licenses that enable identification of individuals.",

    'Intellectual property violation': 'Depictions of artworks, photographs, illustrations, characters or other protected visual works associated with well-known individuals.',
}

def fair_query(model_name, caption=None):
    return_str = IMAGE_SPECIAL_TOKEN[model_name]
    if caption:
        return_str += caption
    instruction = f'{FAIR_INSTRUCTION}{START_FAIR_SYS}{FAIR_SYSMSG}{END_FAIR_SYS}'
    return_str += f'{START_INST} {instruction}{END_INST}'
    return return_str

def safe_query(model_name, categories=list(safe_cat_mapping.keys()), caption=None):
    return_str = IMAGE_SPECIAL_TOKEN[model_name]
    if caption:
        return_str += caption
    # random.shuffle(categories) ############# change llj #############

    category_prompt = ''
    for idx, cat in enumerate(categories):
        category_prompt += f'O{idx+1}: {cat} - '
        category_prompt += safe_cat_mapping[cat] + '\n'
    instruction = f'{SAFE_INSTRUCTION}{START_SAFE_SYS}{category_prompt}{END_SAFE_SYS}'
    return_str += f'{START_INST} {instruction}{END_INST}'
    return return_str


conv_dict = {
    'Internlm': conv2text,
    'Idefics2': idefics_conv2text,
}

def get_conv_func(model_name):
    return conv_dict[model_name]

if __name__ == '__main__':
    print(fair_query('Internlm'))
    print(safe_query('Internlm'))