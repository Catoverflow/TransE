from data import load_data, load_entity_relation
from transE import transE


def main():
    train, test, dev = load_data(
        'data/train.txt', 'data/test.txt', 'data/dev.txt')  # only train is used
    load_trained_model = False
    if load_trained_model:
        # these parameters can be left empty when filter = off
        trans = transE(None, None, train)
        trans.load("data/trained_model")  # load model from file
    else:
        train, test, dev = load_data(
            'data/train.txt', 'data/test.txt', 'data/dev.txt')  # only train is used
        entities = load_entity_relation('data/entity_with_text.txt')
        relations = load_entity_relation('data/relation_with_text.txt')
        trans = transE(entities, relations, train, 170, 0.01, 1)
        trans.emb_init()
        trans.train(2000, 300)
        trans.save("data/trained_model")
    print(trans.hit(dev, n=5, filter=True))  # use dev data to test hit@n value


if __name__ == "__main__":
    main()
