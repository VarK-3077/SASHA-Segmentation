# Separability study

Which of the four crop types a tile selector has to route — normal, tumour core, lesion
boundary, small lesion — can be told apart from a single crop's embedding, and how does that
depend on the pyramid level the crop is read at and on the backbone? Every crop is 256 px at
its own level (footprint 2048/1024/512/256 px at level 0), so per-crop compute is constant
and only the number of crops changes with level.

Buckets come from the lesion xmls rasterised at level 4 (`buckets.py`): core = tumour fraction
>= 0.9; boundary = partial crop touching a lesion larger than the crop footprint; small =
partial crop whose lesions all fit inside the footprint. Recall is also reported by lesion
major axis (ITC < 0.2 mm, micro 0.2-2 mm, macro > 2 mm).

Crops are a stratified per-slide sample (`sample.py`), with half of the normals on tumour
slides drawn within two footprints of a lesion. Normals from the 20 non-exhaustively
annotated training slides are dropped. Split is the repo's `split_4.json` (slide-grouped);
probes are selected on val macro-recall and reported on the 129 test slides.

## How to run (server)

```
bash experiments/sep/rjob.sh cuda:1      # labels -> sample -> extract per level -> probes -> report
```
Everything lands in `/data2/venkatavks/sep/` (`labels/`, `crops_L*.npz`, `feats_L*.h5`,
`probes/`, `report.md`, `lesions.csv`).
