from data import load_data, load_entity_relation
from transE import transE

def main():
    train, test, dev = load_data('data/train.txt','data/test.txt','data/dev.txt') # only train is used
    # entities = load_entity_relation('data/entity_with_text.txt')
    # relations = load_entity_relation('data/relation_with_text.txt')
    # trans = transE(entities, relations, train, 100, 0.01, 1)
    # trans.emb_init()
    # trans.train(400000)
    # save model
    # trans.save("data/trained_model")
    # to load model from file:
    trans = transE(None,None,None) # <- these parameters can be leave empty
    trans.load("data/trained_model")
    # now you can use trans.evaluate(test data)
    # evaluate not implemented yet
    print(trans.hit10_classic(dev)) # use dev data to train

if __name__ == "__main__":
    main()