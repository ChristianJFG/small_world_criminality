"""Run validated, reproducible spectral sweeps from TOML or CLI options."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .experiments import run_sweep
from .plots import generate_plots
from .provenance import environment, source_identity, sha256


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_replay_config(path, resolved):
    # TOML scalars/arrays here share JSON syntax. Exact grids supersede p_points.
    keys = ('preset', 'topologies', 'nodes', 'seeds', 'seed_start', 'ring_k', 'output', 'format')
    lines = [f'{key} = {json.dumps(resolved[key])}' for key in keys]
    lines += ['', '[probabilities]']
    lines += [f'{key} = {json.dumps(values)}' for key, values in resolved['probabilities'].items()]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='TOML experiment file; explicit CLI values override it')
    parser.add_argument('--topologies', nargs='+', choices=['ring', 'square', 'triangular'])
    parser.add_argument('--preset', choices=['quick', 'manuscript'])
    parser.add_argument('--nodes', type=int)
    parser.add_argument('--seeds', type=int, help='number of graph realizations per probability')
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--ring-k', type=int)
    parser.add_argument('--p-points', type=int, help='quick points per region (ring/triangular) or total (square)')
    parser.add_argument('--output')
    parser.add_argument('--format', choices=['png', 'svg', 'pdf'])
    parser.add_argument('--plot-only', action='store_true', help='replot saved modes.csv; retain original run metadata')
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in ('config', 'plot_only')}
    try:
        config = load_config(args.config, overrides)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    resolved = config.resolved()
    base = Path(config.output)
    env = environment()
    source = source_identity()
    for topology in config.topologies:
        out = base / topology
        out.mkdir(parents=True, exist_ok=True)
        if args.plot_only:
            modes = pd.read_csv(out / 'modes.csv')
            files = generate_plots(modes, topology, out, config.format)
            write_json(out / 'replot_metadata.json', {
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'input_modes_sha256': sha256(out / 'modes.csv'),
                'format': config.format, 'source': source, 'environment': env,
                'output_sha256': {p.name: sha256(p) for p in files},
            })
        else:
            metadata = {
                'schema_version': 1, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'topology': topology, 'config': resolved,
                'config_file_sha256': sha256(args.config) if args.config else None,
                'source': source, 'environment': env,
                'provenance_note': 'Recomputed from notebook logic; archival raw tables and environments were not supplied.',
            }
            # A failed rerun cannot leave an old successful metadata record.
            write_json(out / 'metadata.json', metadata)
            write_json(out / 'resolved_config.json', resolved)
            write_replay_config(out / 'resolved_config.toml', resolved)
            try:
                graphs, modes = run_sweep(
                    topology, preset=config.preset, n=config.nodes, seeds=config.seed_count,
                    p_points=config.p_points, ring_k=config.ring_k, seed_start=config.seed_start,
                    p_values=config.probabilities.get(topology), progress=True)
                graphs.to_csv(out / 'graphs.csv', index=False)
                modes.to_csv(out / 'modes.csv', index=False)
                files = generate_plots(modes, topology, out, config.format)
                metadata.update(status='complete', graphs=len(graphs), modes=len(modes),
                                completed_utc=datetime.now(timezone.utc).isoformat(),
                                output_sha256={p.name: sha256(p) for p in [out / 'graphs.csv', out / 'modes.csv', *files]})
            except Exception as exc:
                metadata.update(status='failed', error=str(exc))
                write_json(out / 'metadata.json', metadata)
                raise
            write_json(out / 'metadata.json', metadata)
        print(f'Wrote {len(modes)} mode rows and {len(files)} plots for {topology}', flush=True)


if __name__ == '__main__':
    main()
