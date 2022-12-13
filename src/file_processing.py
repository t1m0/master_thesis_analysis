def extract_age_group(subject):
    if(subject.startswith('30-')):
        age_group = 30
    elif(subject.startswith('50-')):
        age_group = 50
    else:
        age_group = 0
    return age_group