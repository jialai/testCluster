# Importing libraries
import metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import string
import re
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# stop = set(stopwords.words('english'))
# exclude = set(string.punctuation)
# lemma = WordNetLemmatizer()
#
# # Cleaning the text sentences so that punctuation marks, stop words & digits are removed
# def clean(doc):
#     stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
#     punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
#     normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
#     processed = re.sub(r"\d+","",normalized)
#     y = processed.split()
#     return y
#
#
#
path = "PsessionSave.csv"

train_clean_sentences = []
fp = open(path,'r')
for line in fp:
    line = line.strip()
    # cleaned = clean(line)
    # cleaned = ' '.join(cleaned)
    train_clean_sentences.append(line)
#
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(train_clean_sentences)
# modelkmeans = KMeans(n_clusters=10, init='k-means++', max_iter=200, n_init=100)
# modelkmeans.fit(X)
# test_sentences = ["getSurveyInformation,getSurveyServiceInformation,getSurveyQuestionnaireInformation,getSurveyQuestionnaireAnswerInformationForUser,getSurveyServiceInformation,getSurveyQuestionnaireInformation,getSurveyQuestionnaireAnswerInformationForUser,revokeQuestionnaireAnswer,getSurveyServiceInformation,getSurveyQuestionnaireInformation,getSurveyCommunityInformation,getSurveyQuestionnaireAnswerInformationForUser,",\
#                   "instantiateContext,loadProjectBySID,getMediaUris,getMediaId,getStartMediaNodes,getMediaTextAnnotation,getSuccessors,getMediaTextAnnotation,getMediaDescriptionInformationSet,getMediaId,getMediaIds,instantiateContext,",\
#                   "getMediaTextAnnotation,loadProjectBySID,getMediaId,getMediaTextAnnotation,getMediaId,getMediaTextAnnotation,getMediaUris,getMediaId,getMediaTextAnnotation,instantiateContext,getMediaId,getStartMediaNodes,getMediaTextAnnotation,getMediaId,getMediaTextAnnotation,getMediaId,getMediaTextAnnotation,getMediaIds,getStartMediaNodes,getMediaTextAnnotation"]
# Test = vectorizer.transform(test_sentences)
# predicted_labels_kmeans = modelkmeans.predict(Test)
# print ("\n",test_sentences[0],":",predicted_labels_kmeans[0],\
#         "\n",test_sentences[1],":",predicted_labels_kmeans[1],\
#         "\n",test_sentences[2],":",predicted_labels_kmeans[2])



# from tools.labelMap.labelText import LabelText
#
# label = modelkmeans.labels_
# ori_path = "PsessionSave.csv"
# labelAndText = LabelText(label, ori_path)
# labelAndText.sortByLabel(write=True)

######################################################################
#可视化
'''
    2、计算tf-idf设为权重
'''

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(train_clean_sentences))

''' 
    3、获取词袋模型中的所有词语特征
    如果特征数量非常多的情况下可以按照权重降维
'''

word = vectorizer.get_feature_names()
print("word feature length: {}".format(len(word)))

''' 
    4、导出权重，到这边就实现了将文字向量化的过程，矩阵中的每一行就是一个文档的向量表示
'''
tfidf_weight = tfidf.toarray()
print("权重")
print(tfidf_weight)

# # 指定分成7个类
# kmeans = KMeans(n_clusters=40)
# kmodel=kmeans.fit(tfidf_weight)
#
# '''
# 6、可视化
# '''
#
# # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
# tsne = TSNE(n_components=2)
# decomposition_data = tsne.fit_transform(tfidf_weight)

# x = []
# y = []
#
# for i in decomposition_data:
#     x.append(i[0])
#     y.append(i[1])
#
# fig = plt.figure(figsize=(10, 10))
# ax = plt.axes()
# plt.scatter(x, y, c=kmodel.labels_, marker="x")
# plt.xticks(())
# plt.yticks(())
# # plt.show()
# plt.savefig('./test.png', aspect=1)

# metrics.silhouette_score(tfidf_weight, kmodel.labels_, metric='euclidean')

SSE = []  # 存放每次结果的误差平方和
for k in range(1,20):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(tfidf_weight)
    SSE.append(estimator.inertia_)
X = range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.savefig('./test.png', aspect=1)
