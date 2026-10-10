"""Compare daily, fixed-band and 50-candidate grid search using a saved AI setup."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from market import Config
from market_jumps import JumpParameters, build_jump_sets
from portfolio import Policy
from risk import Limits
from search import optimize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,help='Saved portfolio_agent_realworld JSON.')
    parser.add_argument('--stress-profile',choices=('simple','mild','severe'),default=None,
                        help='Default: match the saved run. Override only for a new experiment.')
    parser.add_argument('--output',default='artifacts/spy-baselines.json')
    args=parser.parse_args()
    source=json.loads(Path(args.report).read_text())
    config=Config(**source['config']);limits=Limits(**source['limits'])
    params=JumpParameters(**source['market_diagnostics']['base']['parameters'])
    profile=args.stress_profile or source.get('stress_profile','severe')
    mode='base' if source['scenarios']==['base'] else 'stress'
    paths=source['paths_per_set'];seed=source['seed']
    sets,diagnostics=build_jump_sets(config,paths,seed,mode,params,profile)
    # Old reports may omit the top-level profile; do not silently compare wrong markets.
    if args.stress_profile is None:
        for name,d in diagnostics.items():
            if name not in source['market_diagnostics'] or d['parameters'] != source['market_diagnostics'][name]['parameters']:
                raise ValueError('Cannot match saved scenarios. Specify --stress-profile explicitly.')
    test_paths=source.get('final_test',{}).get('paths_per_set',paths) if source.get('final_test') else paths
    fresh,_=build_jump_sets(config,test_paths,seed+1000000,mode,params,profile)
    methods={'daily':[Policy(0,0)],'fixed_band_0.10':[Policy(.10,0)],
             'grid_50':[Policy(b,k) for b in (.01,.03,.05,.075,.10,.15,.20,.30,.40,.50)
                        for k in (0,2,5,10,20)]}
    results={}
    for name,policies in methods.items():
        best,records=optimize(policies,sets,limits)
        selected=best if name=='grid_50' else records[0]
        final=None
        if selected:
            _,checked=optimize([selected['policy']],fresh,limits)
            final=checked[0]
        results[name]=dict(evaluations=len(records),feasible_candidates=sum(r['feasible'] for r in records),
                           selected=selected,final_test=final,records=records)
        print(name, 'feasible:',sum(r['feasible'] for r in records),'/',len(records),
              'selected fees:',selected['objective'] if selected else None,
              'fresh pass:',final['feasible'] if final else None)
    payload=dict(source=args.report,config=asdict(config),limits=asdict(limits),
                 stress_profile=profile,seed=seed,paths_per_set=paths,
                 market_diagnostics=diagnostics,methods=results,
                 note='AI comparisons require identical scenario/profile/config and evaluation budgets. '
                      'Grid uses 50 unique evaluations; rejected AI proposals do not count as evaluations.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,default=asdict,indent=2)+'\n')
    print('Results:',output.resolve())


if __name__=='__main__':main()
