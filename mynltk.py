import numpy as np
from underthesea import word_tokenize

vietnamese_stopwords = [
    "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa", "chuyện", "có", 
    "có_thể", "cứ", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến_nỗi", "đều", "điều", "do", "đó", 
    "được", "dưới", "gì", "khi", "khoảng", "không", "là", "lại", "lên", "lúc", "mà", "mặc_dù", "một_cách", 
    "này", "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", "qua", "ra", "rằng", 
    "rằng", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên", "trước", "từ", "từng", "và", 
    "vẫn", "vào", "vậy", "vì", "việc", "với", "vừa"
]

def tokenize(sentence):
    """
    Split sentence into an array of words/tokens.
    Filters out stopwords to retain meaningful tokens.
    """
    # Tách từ và loại bỏ stopwords
    tokens = word_tokenize(sentence)
    filtered_tokens = [word for word in tokens if word not in vietnamese_stopwords]
    return filtered_tokens

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    # Tạo mảng chứa các số 0
    bag = np.zeros(len(words), dtype=np.float32)

    # Kiểm tra sự xuất hiện của từng từ trong words và cập nhật vào bag
    for idx, w in enumerate(words):
        if w in tokenized_sentence: 
            bag[idx] = 1

    return bag
