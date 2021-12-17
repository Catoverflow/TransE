import math
import pickle
from os import stat
import time
import random
import numpy as np
from numpy.core.einsumfunc import einsum_path
from numpy.random.mtrand import rand

from data import load_entity_relation


class transE():
    def __init__(self, entity: list, relation: list, triple_rel: list, dim: int = 100, lr: float = 0.01, margin: float = 1) -> None:
        #self.entities = entity
        #self.relations = relation
        self.entities = {}
        self.relations = {}
        self.triple_rels = triple_rel
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.loss = 0
        print("transE object initalized")

    def emb_init(self) -> None:
        ceil = 6/math.sqrt(self.dim)
        self.relations = {relation: transE.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for relation in self.relations}
        self.entities = {entity: transE.norm(
            np.random.uniform(-ceil, ceil, self.dim)) for entity in self.entities}
        # because there are entities/relations not given in entity_with_text nor relation_with_text
        for triple_rel in self.triple_rels:
            if triple_rel[0] not in self.entities:
                self.entities[triple_rel[0]] = transE.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[1] not in self.relations:
                self.relations[triple_rel[1]] = transE.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
            if triple_rel[2] not in self.entities:
                self.entities[triple_rel[2]] = transE.norm(
                    np.random.uniform(-ceil, ceil, self.dim))
        print("transE embedding initalizd")

    @staticmethod
    def dist_l1(h: np.array, r: np.array, t: np.array) -> float:
        return np.sum(np.fabs(h+r-t))

    @staticmethod
    def dist_l2(h: np.array, r: np.array, t: np.array) -> float:
        return np.sum(np.square(h+r-t))

    @staticmethod
    def norm(vector: np.array) -> np.array:
        return vector/np.linalg.norm(vector, ord=2)

    def corrupt(self, head, tail):
        if random.randint(0, 1):
            fake_head = head
            while fake_head == head:  # prevent from sampling the right one
                fake_head = random.sample(self.entities.keys(), 1)[0]
            return fake_head, tail
        else:
            fake_tail = tail
            while fake_tail == tail:
                fake_tail = random.sample(self.entities.keys(), 1)[0]
            return head, fake_tail

    def train(self, eprochs) -> None:
        # here we use Stochastic Gradient Descent.
        print(f"transE training, batch size: 1, eproch: {eprochs}")
        start_timer = time.time()
        for epoch in range(eprochs):
            if epoch % 400 == 0:
                print(
                    f"eproch: {epoch}, loss: {self.loss}, time: {time.time()-start_timer}")
                start_timer = time.time()
                self.loss = 0
            rel_triple = random.sample(self.triple_rels, 1)[0]
            rel_triple.extend(list(self.corrupt(rel_triple[0], rel_triple[2])))
            self.update_embedding(rel_triple, self.dist_l2)

    def update_embedding(self, rel_batch, dist=dist_l2) -> None:
        # sometimes the random sample above will return list with 5 elements (should be 3)
        rel_head, relation, rel_tail, corr_head, corr_tail = rel_batch[:5]
        rel_dist = dist(
            self.entities[rel_head], self.relations[relation], self.entities[rel_tail])
        corr_dist = dist(
            self.entities[corr_head], self.relations[relation], self.entities[corr_tail])
        # hinge loss
        loss = rel_dist-corr_dist+self.margin
        if loss > 0:
            self.loss += loss
            grad_pos = 2 * \
                (self.entities[rel_head] +
                 self.relations[relation]-self.entities[rel_tail])
            grad_neg = 2 * \
                (self.entities[corr_head] +
                 self.relations[relation]-self.entities[corr_tail])

            # update
            grad_pos *= self.lr
            self.entities[rel_head] -= grad_pos
            self.entities[rel_tail] += grad_pos

            # head entity replaced
            grad_neg *= self.lr
            if corr_head == rel_head:  # move away from wrong relationships
                self.entities[rel_head] += grad_neg
                self.entities[corr_tail] -= grad_neg
            # tail entity replaced
            else:
                self.entities[corr_head] += grad_neg
                self.entities[rel_tail] -= grad_neg

            # relation update
            self.relations[relation] -= grad_pos
            self.relations[relation] += grad_neg

    def save(self, filename):
        data = [self.entities, self.relations]
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.entities, self.relations = data

    # the classic way is pretty slow due to enormous distance calculations
    def hit10_classic(self,testdata) -> float:
        hit10 = 0
        count = 0
        for head, rel, tail in testdata:
            if count % 20 == 0:
                print(f"{count}/{len(testdata)} cases evaluated, hit10 sum: {hit10}")
            assume_tail = self.entities[head] + self.relations[rel]
            result = {np.sum(np.square(assume_tail - self.entities[entity])):entity for entity in self.entities.keys()}
            result = dict(sorted(result.items())[:10])
            if tail in result.values():
                hit10 += 1
            count += 1
        hit10 /= len(testdata)
        return hit10