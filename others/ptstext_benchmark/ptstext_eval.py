from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class Ptstext_EvalCap:
    def __init__(self, ptstext_prediction, ptstext_gt):
        self.evalPcds = []
        self.eval = {}
        self.pcdToEval = {}
        self.ptstext_prediction = ptstext_prediction
        self.ptstext_gt = ptstext_gt
        self.params = {'pcd_id': ptstext_prediction.getPcdIds()}

    def evaluate(self):
        
        # open the proxy in the server
        import subprocess
        subprocess.run(['proxy_on'], shell=True)

        pcdIds = self.params['pcd_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for pcdId in pcdIds:
            gts[pcdId] = self.ptstext_prediction.pcdToAnns[pcdId]
            res[pcdId] = self.ptstext_gt.pcdToAnns[pcdId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setPcdToEvalPcds(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setPcdToEvalPcds(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalPcds()

    def setEval(self, score, method):
        self.eval[method] = score

    def setPcdToEvalPcds(self, scores, pcdIds, method):
        for pcdId, score in zip(pcdIds, scores):
            if not pcdId in self.pcdToEval:
                self.pcdToEval[pcdId] = {}
                self.pcdToEval[pcdId]["pcd_id"] = pcdId
            self.pcdToEval[pcdId][method] = score

    def setEvalPcds(self):
        self.evalPcds = [eval for pcdId, eval in self.pcdToEval.items()]


