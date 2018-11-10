import hyperdimensionalsemanticspace
from nltk import sent_tokenize
from nltk import word_tokenize
import simpletextfilereader

# for each text
# compare polar terms with canonical terms in vector space
# establish if polar opposites have systematic correlation with canonical opposites

canonicalgood = ["good", "alive"]
canonicalbad = ["bad", "dead"]
canonicals = canonicalbad + canonicalgood
probegood = ["easy"]
probebad = ["difficult"]
probes = probebad + probegood
items = canonicals + probes

# for each polar term, build a utterance context vector

dimensionality = 2000
denseness = 10
contextspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness, "polarcanonical")
utterancespace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness, "polarcanonical")

#simpletextfilereader.readstats()
window = 2
files = simpletextfilereader.getfilelist()
i = 0
antalsatser = 0
antalord = 0
threshold = 0.1
for file in files:
    i += 1
    texts = simpletextfilereader.doonejsontextfile(file)
    flag = []
    for text in texts:
        ss = sent_tokenize(text.lower())
        for sent in ss:
            words = word_tokenize(sent)
            i = 0
            for word in words:
                i += 1
                if word in items:
                    flag = flag.append(word)
                    contextspace.observe(word)
                    lhs = words[i-window:i]
                    rhs = words[i+1:i+window+1]
                    for lw in lhs:
                        contextspace.addintoitem(word, lw, simpletextfilereader.weight(word), "before")
                    for rw in rhs:
                        contextspace.addintoitem(word, rw, simpletextfilereader.weight(word), "after")
            if len(flag) >= 1:
                antalord += len(words)
                antalsatser += 1
            for knownword in flag:
                utterancespace.observe(knownword)
                for word in words:
                    utterancespace.addintoitem(knownword, word)
            flag = []

    if i > 3:
        i = 0
        print("==", antalsatser, antalord, "============", sep="\t")
        for s in [contextspace, utterancespace]:
            print(s.name, "----------", sep="\t")
            tokenneighbours = {}
            nabesize = {}
            for oneitem in s.contextspace:
                tokenneighbours[oneitem] = 0
                nabesize[oneitem] = 0
                nabe = s.contextneighbours(oneitem, 0, True)
                for nnn in nabe:
                    tokenneighbours[s.tag[oneitem]] += nnn[1] / ((len(nabe) + 1) * len(nabe))
                    if nnn[1] > threshold:
                        nabesize[s.tag[oneitem]] += 1 / len(nabe)
            for lemma in tokenneighbours:
                print(lemma, tokenneighbours[lemma], nabesize[lemma], sep="\t")
