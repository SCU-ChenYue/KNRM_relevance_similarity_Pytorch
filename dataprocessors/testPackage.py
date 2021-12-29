from dataprocessors.tokenizer import Segment_jieba
from dataprocessors.embedding import Embedding
from dataprocessors.vocab import Vocab
from dataprocessors.dataset import Dataset
import torch.nn as nn
import torch


embedding = Embedding("../data/pretrainVectors/sgns.baidubaike.bigram-char")
# print(embedding.w2v.word_vec("我"))
query_list = [74, 401, 6, 5290, 20001, 20001, 20001, 2, 1531, 1, 6, 10016, 3310, 20001, 18138, 2, 20001, 3]
print("====================================")
segment = Segment_jieba("")    # 加载停用词
# print(segment.seg("南山乔木大又高，树下不可歇阴凉。汉江之上有游女，想去追求不可能。汉江滔滔宽又广，想要渡过不可能。江水悠悠长又长，乘筏渡过不可能。柴草丛丛错杂生，用刀割取那荆条。"))

vocab = Vocab("../data/example/example_data.txt", segment, embedding)
print(f"vocab length: {len(vocab)}")
# for id in query_list:
#     print(vocab.idx2word[id])
# # print(embedding.w2v.word_vec('地球'))
# print(embedding.w2v.index2word)

# dataset = Dataset("../data/example/example_data.txt", segment, vocab.word2idx, 20, 20)

embed_layer = nn.Embedding(len(embedding.w2v.index2word), embedding.vector_size, padding_idx=0)
embed_layer.weight.data.copy_(torch.from_numpy((embedding.get_vectors)))
# print(vocab.idx2word[74])
# print(embedding.w2v.index2word[74])
# print(embed_layer(torch.tensor([74])))
print("当前测试的字符是。")
print("。的向量是：")
print(embedding.w2v.word_vec("。"))
print("。的id是：")
print(vocab)

