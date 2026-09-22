"""Debug: why does banking77 score 0? Inspect raw Laya answers on 3 holdout rows."""
import laya

from eval.run_eval import load_holdout

agent = laya.load("convaiinnovations/laya")
rows = load_holdout(["banking77"])["banking77"][:3]
for r in rows:
    qid = next(iter(r.questions))
    out = agent.predict(r.state, r.questions)
    ans = out["answers"][qid]
    print("GOLD:", r.gold[qid].value)
    print("PRED:", ans["choice"])
    print("top3:", sorted(((v, k) for k, v in ans["probabilities"].items()), reverse=True)[:3])
    print("crit n:", len(r.questions[qid]["criteria"]), "| state:", str(r.state)[:90])
    print("---")
