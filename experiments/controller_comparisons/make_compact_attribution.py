"""Keep the architecture diagnostic brief and retain its full table in the supplement."""
from pathlib import Path
import re
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.exp_common import regime_bundle


def main():
    generated=ROOT/'manuscript/generated'
    source=(generated/'comparative_reference.tex').read_text(encoding='utf-8')
    tables=re.findall(r'\\begin\{table\}\[tbp\].*?\\end\{table\}',source,flags=re.S)
    assert len(tables)==1
    (generated/'reference_attribution_table.tex').write_text(tables[0]+'\n',encoding='utf-8')
    pairs=pd.read_csv(ROOT/'experiments/results/comparative_reference_pairwise.csv').set_index('fee')
    methods=pd.read_csv(ROOT/'experiments/results/comparative_reference_method_summary.csv')
    reference=methods[methods.method=='reference_value_one_step']
    params,_,_=regime_bundle('confirmatory',['winter_weekday'])
    efc=reference.mean_throughput_kwh/(2*params['winter_weekday'].E_max)
    differences=[float(pairs.loc[f,'mean_diff']) for f in (20,40,80)]
    text=(f'The learned-minus-reference cost differences were ${differences[0]:.1f}$,\n'
          f'${differences[1]:.1f}$, and ${differences[2]:.1f}$ EUR/day at the three fee anchors.\n'
          'These compare the same one-step architecture with and without the neural\n'
          f'continuation correction. The reference used only {efc.min():.2f}--{efc.max():.2f} equivalent\n'
          'full cycles per day. This ablation isolates the learned continuation;\n'
          'the tariff-aware rule provides the practical non-learning comparison.\n'
          'The supplement reports the paired\n'
          'intervals and full cost decomposition.\n')
    (generated/'compact_reference.tex').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':
    main()
