from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk



class Vectorize:
    def __init__(self):
        nltk.download('punkt')
        # define a list of documents.
        self.data = ["This is the first document",
                     "This is the second document",
                     "This is the third document",
                     "This is the fourth document",
                     "The cat jumped over the lazy dog the quick brown fox jumps over the lazy dog"
                     ]
        # preprocess the documents, and create TaggedDocuments
        self.tagged_data = [
            TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(self.data)
        ]
        self.train()

    # train the model
    def train(self):
        # train the Doc2vec model
        self.model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
        self.model.build_vocab(self.tagged_data)
        self.model.train(self.tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
    #  predict the vectors for the documents
    def predict(self,data):
        self.document_vectors = self.model.infer_vector(word_tokenize(data.lower()))
        return self.document_vectors

    def print_vectors(self):
        # print the document vectors
        self.train()
        self.predict()
def main():
    vectorMaker = Vectorize()
    result = vectorMaker.predict("This is the fifth document")
    print(result)

if __name__ == '__main__':
    main()
