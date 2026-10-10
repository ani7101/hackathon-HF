"""Simulator-only reference search; no language model is used."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hedgesim import PathSet, default_params, best_policy
start=time.perf_counter()
ps=PathSet(default_params(),2000,101)
result={'mode':'level-5 simulator reference search; no model', 'paths':2000, 'seed':101,'parameters':ps.prm,'daily':ps.run(), 'unhedged':ps.unhedged()}
for family in ['interval','band','formula']:
    j,policy,summary=best_policy(ps,family)
    result[family]={'J':j,'policy':policy,'summary':summary}
result['seconds']=time.perf_counter()-start
print(json.dumps(result,indent=2))
