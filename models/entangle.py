
import numpy as np

# input : E : E[i,j] = entangleness between token i and token j
def minimum_entangle(E):
    l,l_w = E.shape
    assert(l == l_w)  # This should be equal because we only consider intra entangle for now.

    D = dict()  # D[i,j] = minimum loss for grouping token[0:i] with j groups
    last_begin = dict()  # history[i,j] = begin of last group for min_grouping(token[0:i])

    inf = 1e30
    D[(0,0)] = 0  # Base case

    for i in range(1, l+1):
        D[(i,0)] = inf
        for n_group in range(1, i+1):
            # i elem, n_group,

            min_loss = inf
            pre_idx = -1
            for last_group_size in range(1,i-n_group+2):
                idx = i - last_group_size  # end of previous group (exclusive)
                D_prior = D[(idx, n_group-1)]
                loss_added = sum([E[i1, i2] for i1 in range(0,idx) for i2 in range(idx, i)])
                loss = D_prior + loss_added
                if min_loss > loss:
                    min_loss = loss
                    pre_idx = idx

            D[(i, n_group)] = min_loss
            last_begin[(i, n_group)] = pre_idx

        assert(D[(i, 1)] == 0)

    return Deentangler(D, last_begin, l)


class Deentangler:
    def __init__(self, D, last_begin, l):
        self.D = D
        self.last_begin = last_begin
        self.l = l

    def loss_group(self):
        r = []
        for j in range(1, self.l):
            loss = self.D[(self.l, j)]
            r.append((j, loss))
        return r

    def group_by_loss(self, maximum_loss):
        max_group = 0
        for j in range(1,self.l):
            if self.D[(self.l, j)] < maximum_loss:
                max_group = j

        return self.trace_group(max_group)

    def group_by_count(self, minimum_group):
        return self.trace_group(minimum_group)

    def trace_group(self, n_group):
        l = self.trace(n_group) + [self.l]
        return zip(l[:-1], l[1:])

    # output : list[group_beginning] , length = n_group
    def trace(self, n_group):
        begin_list = []
        end_idx = self.l
        while end_idx > 0 :
            begin = self.last_begin[(end_idx,n_group)]
            begin_list.append(begin)
            end_idx = begin
            n_group = n_group -1

            assert(n_group >= 0)
        begin_list.reverse()
        return begin_list









