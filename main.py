from pickle import TRUE
from data import load_data, load_entity_relation
from transE import transE
from transH import transH
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Train and test in TransH model')
    parser.add_argument('eproach', type=int, default = 2000)
    parser.add_argument('batch', type=int, default = 300)
    parser.add_argument('dimention', type=int, default = 200)
    parser.add_argument('-L', help='Load the trained model', default=False)
    parser.add_argument('-S', help='Save the trained model', default=False)
    args = parser.parse_args()
    train, test, dev = load_data(
        'data/train.txt', 'data/test.txt', 'data/dev.txt')  # only train is used
    if args.L:
        # these parameters can be left empty when filter = off
        trans = transH(None, None, train)
        trans.load("data/trained_model")  # load model from file
    else:
        train, test, dev = load_data(
            'data/train.txt', 'data/test.txt', 'data/dev.txt')  # only train is used
        entities = load_entity_relation('data/entity_with_text.txt')
        relations = load_entity_relation('data/relation_with_text.txt')
        trans = transH(entities, relations, train, args.dimention, 0.01, 1)
        trans.emb_init()
        trans.train(args.eproach, args.batch)
        if args.S:
            trans.save("data/trained_model")
    print(trans.hit(dev, n=5, filter=True))  # use dev data to test hit@n value


if __name__ == "__main__":
    main()
