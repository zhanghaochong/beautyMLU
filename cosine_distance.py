import sys
import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)

def load_data(filepath):
    data=[]
    with open(filepath,"r") as f:
        for i in f.readlines():
            data.append(float(i.strip()))
    return np.array(data)
    
if __name__ == '__main__':
    print("cosine_similarity:{}".format(cosine_similarity(load_data(sys.argv[1]),load_data(sys.argv[2]))))