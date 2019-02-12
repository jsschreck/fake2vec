import pickle, sys, numpy as np
from gensim.models import Doc2Vec
from keras import backend as K

class Fake2Vec:

	def __init__(self, infer_steps = 20, datapath = './'):

		self.infer_steps = infer_steps
		self.datapath = datapath

		with open(self.datapath + "models/classifier/model.h5.pkl", "rb") as fid:
			self.classifier = pickle.load(fid)

		with open(self.datapath + "data/label_encoders.pkl", "rb") as fid:
			self.encoders = pickle.load(fid)

		DOC2VEC_MODEL = self.datapath + "models/doc2vec/doc2vec-VecSize-1000_MinDoc-10_Window-300.model"
		self.doc2vec = Doc2Vec.load(DOC2VEC_MODEL)
		self.vecsize = self.doc2vec.docvecs[0].shape[0]

		self.fact_decode = self.encoders[1].inverse_transform(range(3))
		self.bias_decode = self.encoders[2].inverse_transform(range(7))

	def predict(self, X, topk = 5):
		X = self.doc2vec.infer_vector(X, steps=self.infer_steps).reshape(-1,self.vecsize)
		pub, fact, bias = self.classifier.predict(X)
		top_pubs = np.argsort(pub[0])[-topk:][::-1]
		pubs_decode = [[self.encoders[0].inverse_transform([k])[0], float("%0.2f" % (100*pub[0][k]))] for k in top_pubs]
		facts = [[decoded, float("%0.2f" % (100*fact_score))] for fact_score, decoded in sorted(zip(fact[0],self.fact_decode))[::-1]]
		affiliation = [[decoded, float("%0.2f" % (100*bias_score))] for bias_score, decoded in sorted(zip(bias[0],self.bias_decode))[::-1]]
		return pubs_decode, facts, affiliation

def main(X, infer_steps = 20, path = '../'):
	model = Fake2Vec(infer_steps = infer_steps, datapath = path)
	result = model.predict(X)
	K.clear_session()
	return result

if __name__ == '__main__':

	X = str(sys.argv[1])
	infer_steps = 20

	model = Fake2Vec(infer_steps = infer_steps)
	pubs_decode, facts, affiliation = model.predict(X)

	print("------------------------------------------------")
	print("Document similar to publisher")
	for (decoded,pub_score) in pubs_decode:
		print("...", decoded, ":", pub_score)

	print("------------------------------------------------")
	print("Factual assesment of document:")
	for (decoded,fact_score) in facts:
		print("...", decoded, ":", fact_score)

	print("------------------------------------------------")
	print("Lean/affiliation of document:")
	for (decoded,bias_score) in affiliation:
		print("...", decoded, ":", bias_score)
