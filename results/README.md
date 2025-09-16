# Results Directory

Each optimization run writes a timestamped folder to this directory.  Folders
use the pattern `YYYYMMDD-HHMMSS_mmm__<run-name>` so they sort
chronologically, e.g. `20250913-175823_962__x35_air`.

## Standard Run Layout
```
<run>/
├── artifacts/
│   └── result.json        # Serialized Params/Result/metadata for the run
├── figures/               # Radius, slope, objective, and volume plots
├── logs/
│   └── run.log            # Combined console/file log from bin/run_opt.py
├── meta/
│   └── config.json        # CLI arguments, git info, environment snapshot
├── tables/
│   ├── summary.csv        # Parameter summary (created by plot_results.py)
│   └── summary.txt        # Human-readable summary table
├── reports/               # summary.html, summary.md, optional summary.pdf
└── manifest.json          # Checksums for every file in the run folder
```

The manifest and metadata are created automatically by `airship_opt.io_viz`
when `bin/run_opt.py` finishes and when additional tooling updates the run.

## Latest Pointers
The driver maintains a helper pointer after every run:
- `latest__<run-name>` – symlink on Unix platforms
- `latest__<run-name>.txt` – text file with an absolute path (Windows fallback)

You can pass either pointer to the helper scripts:
```bash
# Resolve canonical locations for the latest "x35" run
python bin/latest.py --run-name x35 --json

# Plot using the pointer instead of the full path
python bin/plot_results.py --result results/latest__x35.txt

# Build reports (HTML/MD, optionally PDF)
python bin/make_report.py --result results/latest__x35.txt --no-pdf
```

`bin/plot_results.py` will back-fill `tables/` and refresh the manifest so the
run directory is self-describing even if you copy it elsewhere.

## Housekeeping Notes
- Large artifacts are not committed to source control; prune as needed.
- Keep raw run folders if you want reproducible provenance—`meta/config.json`
  records CLI arguments, git SHA, and relevant environment details.
- Pointers are refreshed automatically by `bin/run_opt.py`. To retarget one
  manually, recreate the symlink (or update the `.txt` file on Windows).
