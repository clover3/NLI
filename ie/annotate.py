
import pickle
import csv

def dump_plain_str():
    path = "../data/multinli_0.9/nli explain.csv"
    reader = csv.reader(open(path, "r"), delimiter=',')

    f = open("dump.txt", "w")
    for idx, row in enumerate(reader):
        if idx ==0 : continue
        premise = row[0]
        hypothesis= row[1]
        tokens_premise = row[2].split()
        tokens_hypothesis= row[3].split()

        for t in tokens_hypothesis:
            if t.lower() not in hypothesis.lower():
                raise Exception(t)
        for t in tokens_premise:
            if t.lower() not in premise.lower():
                print(premise)
                raise Exception(t)

        f.write(row[0] + "\n")
        f.write(row[1] + "\n")
    f.close()

def visualize():

    result = pickle.load(open("../pickle/match_intgrad", "rb"))


    f= open("../match_intgrad.html", "w")
    f.write("<html>")

    def print_color_html(word, r):
        r = 255 - r
        bg_color =  "ff" + ("%02x" % r) + ("%02x" % r)

        html = "<td bgcolor=\"#{}\">&nbsp;{}&nbsp;</td>".format(bg_color, word)
        #    html = "<td>&nbsp;{}&nbsp;</td>".format(word)
        return html
    f.write("<body>")
    f.write("<div width=\"400\">")
    for entry in result:
        pred_p, pred_h, prem, hypo = entry

        max_score = max(pred_p[0][0], pred_h[0][0])
        p_score = {idx:score for score, idx in pred_p}
        h_score = {idx:score for score, idx in pred_h}
        print(max_score)
        cut = max_score * 0.5
        print(cut)

        f.write("<tr>")
        f.write("<td><b>Premise<b></td>\n")
        f.write("<table style=\"border:1px solid\">")
        for i, token in enumerate(prem):
            print("{}({}) ".format(token, p_score[i]), end = "")
            #r = 100 if p_score[i] > cut else 255
            r = int(p_score[i] * 255 / max_score)
            r = r if r > 0 else 0
            f.write(print_color_html(token, r))
        print()
        f.write("</tr>")
        f.write("</tr></table>")

        f.write("<td><b>Hypothesis<b></td>\n")
        f.write("<table style=\"border:1px solid\">")
        for i, token in enumerate(hypo):
            print("{}({}) ".format(token, h_score[i]), end = "")
            r = 100 if h_score[i] > cut else 255
            r = int(h_score[i] * 255 / max_score)
            r = r if r > 0 else 0
            f.write(print_color_html(token, r))
        print()
        f.write("</tr>")

        f.write("</tr></table>")
        f.write("</div><hr>")

    f.write("</div>")
    f.write("</body>")
    f.write("</html>")


visualize()