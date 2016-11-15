#!/usr/bin/env python2
from collections import defaultdict


def find_right_bracket(s):
    """
    find the right bracket to complete the first left bracket
    '(DT The)(NN boy)' ->7 
    """
    left = 0
    for i in xrange(len(s)):
        c = s[i]
        if c == '(':
            left += 1
        elif c == ')':
            left -= 1
            if left == 0:
                return i
    return -1


def count(s, counter, total):
    """
    count a node of parsing tree into counter. The input is a string
    representing the tree node, and the counter. Return the type of the node"""
    bracket_index = s.find('(', 2)
    if bracket_index < 0:
        # unary
        space_index = s.find(' ', 2)
        parent = s[1:space_index]
        child = s[space_index + 1:-1]
        counter[parent][child] += 1
        counter[parent]['']
        total[parent] += 1
        return parent

    parent = s[1:bracket_index]
    s = s[bracket_index:-1]
    second_bracket_index = find_right_bracket(s) + 1
    left = s[:second_bracket_index]
    right = s[second_bracket_index:]
    left_type = count(left, counter, total)
    right_type = count(right, counter, total)
    counter[parent][(left_type, right_type)] += 1
    total[parent] += 1
    return parent


def make_model_line(parent, child, p):
    if isinstance(child, tuple):
        child = " ".join(child)
    return parent + ' # ' + child + ' # ' + str(p)


def train(training_data, model_file):
    counter = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    with open(training_data, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            count(line, counter, total)

    output = []
    prob = defaultdict(dict)
    for parent, v in counter.iteritems():
        ftotal = float(total[parent])
        for child, n in v.iteritems():
            if child == '':
                continue
            p = n / ftotal
            prob[parent][child] = p
            output.append(make_model_line(parent, child, p))
    with open(model_file, 'w') as f:
        f.write('\n'.join(sorted(output)))


if __name__ == '__main__':
    model = train('training_data.txt', 'model.txt')
