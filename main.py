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


def count(s, lex_counter, syn_counter, total):
    """
    count a node of parsing tree into counter. The input is a string
    representing the tree node, and the counter. Return the type of the node
    """
    bracket_index = s.find('(', 2)
    if bracket_index < 0:
        # unary
        space_index = s.find(' ', 2)
        parent = s[1:space_index]
        child = s[space_index + 1:-1].lower()
        lex_counter[parent][child] += 1
        total[parent] += 1
        return parent

    parent = s[1:bracket_index]
    s = s[bracket_index:-1]
    second_bracket_index = find_right_bracket(s) + 1
    left = s[:second_bracket_index]
    right = s[second_bracket_index:]
    left_type = count(left, lex_counter, syn_counter, total)
    right_type = count(right, lex_counter, syn_counter, total)
    syn_counter[parent][(left_type, right_type)] += 1
    total[parent] += 1
    return parent


def make_model_line(parent, child, p):
    return parent + ' # ' + child + ' # ' + str(p)


def train(training_data, model_file):
    # count
    lex_counter = defaultdict(lambda: defaultdict(int))
    syn_counter = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    dic = {}
    with open(training_data, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            count(line, lex_counter, syn_counter, total)

    # calc prob and dic
    output = []
    lex_prob = {}
    syn_prob = {}
    for parent, v in lex_counter.iteritems():
        ftotal = float(total[parent])
        for child, n in v.iteritems():
            if child == '':
                continue
            p = n / ftotal
            lex_prob[(parent, child)] = p
            output.append(make_model_line(parent, child, p))
    for parent, v in syn_counter.iteritems():
        ftotal = float(total[parent])
        for child, n in v.iteritems():
            if child == '':
                continue
            p = n / ftotal
            syn_prob[(parent, child[0], child[1])] = p
            output.append(make_model_line(parent, " ".join(child), p))
    with open(model_file, 'w') as f:
        f.write('\n'.join(sorted(output)))

    return lex_prob, syn_prob, dic


def parse(s, lex_prob, syn_prob, dic):
    """
    parsing with CYK algorithm
    """
    s = s.split(' ')
    m = len(s)
    chart = [[defaultdict(list) for x in xrange(m)] for y in xrange(m)]
    for j in xrange(m):
        lower_j = s[j].lower()
        for k, p in lex_prob.iteritems():
            parent = k[0]
            child = k[1]
            if lower_j == child:
                # [parent, left_child, right_child, prob, inside, outside]
                chart[j][j][parent].append([parent, s[j], None, p, None, None])
                break
        for i in xrange(j - 1, -1, -1):
            for k in xrange(i, j):
                for abc, p in syn_prob.iteritems():
                    a = abc[0]
                    b = abc[1]
                    c = abc[2]
                    if b in chart[i][k] and c in chart[k + 1][j]:
                        tree_list_b = chart[i][k][b]
                        tree_list_c = chart[k + 1][j][c]
                        for tree_b in tree_list_b:
                            for tree_c in tree_list_c:
                                p_b = tree_b[3]
                                p_c = tree_c[3]
                                p_a = p * p_b * p_c
                                chart[i][j][a].append(
                                    [a, tree_b, tree_c, p_a, None, None])
    return chart[0][m - 1]['S']


def make_tree_string(tree):
    if tree[2] is None:
        # lex
        return '(' + tree[0] + ' ' + tree[1] + ')'
    left = make_tree_string(tree[1])
    right = make_tree_string(tree[2])
    return '(' + tree[0] + left + right + ')'


def inside_outside(results, f):
    pass


def save_parsed_result(results, parse_file):
    print results
    most_likely = None
    for result in results:
        if most_likely is None or result[3] > most_likely[3]:
            most_likely = result
    f = open(parse_file, 'w')
    f.write(make_tree_string(most_likely) + '\n')
    f.write(str(most_likely[3]) + '\n')
    inside_outside(results, f)
    f.close()


if __name__ == '__main__':
    lex_prob, syn_prob, dic = train('training_data.txt', 'model_file.txt')

    test_str = 'A boy with a telescope saw a girl'
    res = parse(test_str, lex_prob, syn_prob, dic)
    save_parsed_result(res, 'parse_file.txt')
