
import spacy

#import spacy.cli
#spacy.cli.download("en_core_web_sm")
import matplotlib.pyplot as plt
import seaborn as sns

from processing import tf_idf_scores

nlp = spacy.load('en_core_web_sm')

docs = [nlp(text) for text in (t0, t1, t2, t3, t4, t5, t6)]

res = tf_idf_scores(docs)

sns.set()

fig, ax = plt.subplots(figsize = (15,3))
sns.heatmap(res, ax = ax)
plt.savefig('tf_idf_scores.png')