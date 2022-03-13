- TODO:
1. pitch detection algorithm
    - librosa.pyin
    - pysptk
    pysptk 나 pyin 과 같은 pitch detection algorithm을 사용하여 프레임별로 f0 를 추출하고, 원하시는 기준에 맞추어 일정 수준 이상의 값을 고르시는 방법

**Filter type**

**Q factor**

**Bandwidth**

**Gain**

**Phase**

**Equalizer**

1. Parametric EQ

2. Semi-Parametric EQ

3. Dynamic EQ

4. Graphic EQ

5. Shelving EQ

---
# Tree
FilterDesign
    - lib
        - biquad_cookbook.py
        - fi.py
        - filt.py
        - filter_analyze.py
        - filter_application.py
        - graphic_equalizer.py
        - realtimedsp.py
        - util.py
        - wav_generator.py
        - zplane.py
        - c
            - [_filt.c]
        - data
            - audacity_filtered.txt
            - audacity_origin.txt
        - debug
            - log.py
            - logging.json
            - trace_memory.py
            - util.py
    - study
    - main.py
    - pylintrc
    - README.md
    - requirements.txt
    - .style.yapf
    - LICENSE
