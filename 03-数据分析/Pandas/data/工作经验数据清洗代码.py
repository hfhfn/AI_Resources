job.experience.fillna('未知',inplace = True)
def process_experience(x):
    if x in ['1-3年','2年经验','经验1-3年']:
        return '1-3年'
    elif x in ['3-5年','经验3-5年','3-4年经验']:
        return '3-5年'
    elif x in ['1年经验','1年以下','经验1年以下','一年以下','经验应届毕业生','不限','经验不限','无需经验','无经验']:
        return '一年以下/应届生/经验不限'
    elif x in ['5-10年','经验5-10年','5-7年经验','8-9年经验','10年以上经验','10年以上','经验10年以上']:
        return '5年以上'
    else:
        return x
    
job['exp'] = job.experience.apply(process_experience)
job['exp'].value_counts()