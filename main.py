import pytesseract as ocr
import fitz
import vectorize
import os
from sklearn.ensemble import RandomForestClassifier

DATADIR = "data/"
TRAINING_DATA = "training_data/"
TRAINING_LABELS = "training_labels/"
Training_label_file = "training.txt"
TO_PREDICT_DATA = "predict/"


vectorizer = vectorize.Vectorize()

# Vectorize the pdf file data
def read_data(folderName):
    vectors = []
    for page in os.listdir(folderName):
        if page == ".DS_Store":
            continue
        doc = fitz.open(os.path.join(folderName, page))
        pix = doc[0].get_pixmap()
        pix.save("temp.png")
        ocrtxt = ocr.image_to_string("temp.png")
        vectors.append(vectorizer.predict(ocrtxt))
    return vectors

def read_fileData(fileName):
    vectors = []
    doc = fitz.open(fileName)
    pix = doc[0].get_pixmap()
    pix.save("temp.png")
    ocrtxt = ocr.image_to_string("temp.png")
    vectors.append(vectorizer.predict(ocrtxt))
    return vectors

# Read the labels for the pdf file
def read_labels(fileName):
    labels = []
    with open(fileName, 'r') as f:
        labels = f.read().splitlines()
    return labels

# Prints the results for the data in TO_PREDICT_DATA
def main():
    X_train = read_data(DATADIR + TRAINING_DATA)
    y_train = read_labels(DATADIR + TRAINING_LABELS + Training_label_file)
    desiredPrediction = read_labels(DATADIR + TO_PREDICT_DATA)
    
    test_data = read_fileData("6.pdf")
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)
    
    
    print(classifier.predict(desiredPrediction))
  

if __name__ == '__main__':
    main()
