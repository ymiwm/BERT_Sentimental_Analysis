import MeCab
import os

m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')

setnames = ['train', 'test']


def get_pos_tagging_result(result):
    # print(result)
    lines = result.split("\n")[:-2]

    result = []
    for line in lines:
        ms, info = line.split('\t')

        compound_tag, _, first, _, _, _, _, mophms = info.split(',')

        if mophms == "*":
            result.append((ms, compound_tag))
            continue

        for i, mophm in enumerate(mophms.split("+")):
            idx = mophm.rfind("/")
            idx2 = mophm[:idx].rfind("/")

            m = mophm[:idx2]
            t = mophm[idx2 + 1:idx]

            result.append((m, t))

    return result


data_dir = "data"
for setname in setnames:
    import time

    start = time.time()
    f = open(os.path.join(data_dir, "%s.txt" % setname), encoding='utf-8')
    of = open(os.path.join(data_dir, "%s_tagged.txt" % setname), 'w', encoding='utf-8')
    f.readline()  # first line remove

    lines = f.readlines()
    for line in lines:
        line = line.strip()

        doc_id, sent, label = line.split("\t")

        result = get_pos_tagging_result(m.parse(sent))

        for token in result:
            mophmtag = token[0] + "/" + token[1]
            tag = token[1]

            of.write(mophmtag + ' ' + tag + '\n')

        of.write(doc_id + "\t" + label + "\n\n")

    print(setname, time.time() - start)
