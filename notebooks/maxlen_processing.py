# .gitignore

from konlpy.tag import Mecab
from pathlib import Path
import spacy
# 토큰화 함수로 MeCab 사용
spacy_en = spacy.load('en')
tokenize_ko = Mecab().morphs
tokenize_en = lambda text: [tok.text for tok in spacy_en.tokenizer(text)]

p = './data/translation/koen/{}.{}'
svp = './data/translation/koen/{}30.{}'
for filename in ['train', 'dev', 'test']:
    data_ko = Path(p.format(filename, 'ko')).read_text().splitlines()
    data_en = Path(p.format(filename, 'en')).read_text().splitlines()
    temp = []
    for ko, en in zip(data_ko, data_en):
        if len(tokenize_ko(ko)) <= 30 and len(tokenize_en(en)) <= 30:
            temp.append((ko, en))   
    data_ko, data_en = list(zip(*temp))
    Path(svp.format(filename, 'ko')).write_text("\n".join(data_ko))
    Path(svp.format(filename, 'en')).write_text("\n".join(data_en))