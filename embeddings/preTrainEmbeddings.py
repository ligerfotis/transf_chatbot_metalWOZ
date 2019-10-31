import fasttext

def create_model():
    model = fasttext.train_unsupervised('/data/merged.csv', minn=3, maxn=6, dim=300)

    model.save_model("fastnet_model.bin")

def load_model():

    return fasttext.load_model("fastnet_model.bin")


def creatEmbeddingsMatrix():
    pass


def create_vocabulary(fasttext_model, filename="vocab_fasttext.txt"):
    with open(filename, "w") as file:
        for word in fasttext_model.words:
            file.writelines("\'" + word + "\'" + "\n")



#create_model()
model = load_model()
create_vocabulary(model)

print(model.words)   # list of words in dictionary
print(model['αγόρι'])  # get the vector of the word 'king'
