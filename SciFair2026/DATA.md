# Data layout (PrecisionPhage)

## Host genomes (`data/fastas/`)

| Path | In Git | Notes |
|------|--------|--------|
| `hosts/*.fasta` | **Yes** | One RefSeq-derived assembly per host species (source of truth). |
| `host_download_log.csv` | **Yes** | Download status per species. |
| `host_genomes.fasta` | **No** | **Generated** by `run_pipeline.py`: concatenation of `hosts/*.fasta` with species headers. It can exceed GitHub’s file-size limits; clone the repo and run the pipeline (or concatenate locally) to recreate it. |

The pipeline skips re-downloading when `host_download_log.csv` exists and `host_genomes.fasta` is large enough; delete the log file if you need a full re-download.

## Other outputs

Plots, `results/*.csv`, and `raw/` caches are produced by `scripts/run_pipeline.py`. Track or ignore them in `.gitignore` according to what you want versioned.
