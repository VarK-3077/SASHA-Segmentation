#!/bin/bash
# Full level-2 maps of the trusted train slides on vivi GPU 1, for training on segmenter outcomes. Slides
# live on the HDD, which reads a file straight through fine but random crops badly, so a few at a time
# are copied to NVMe, segmented and deleted. Note: the level-2 head was trained on these slides.
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 TORCH_HOME=/data2/venkatavks/torch_cache HF_HOME=/data2/venkatavks/hf_cache TOK_LEVEL=2 TOK_BACKBONE=kaiko_vitl14
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOWSEP_ROOT=/var/tmp/venkatavks/lowsep RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen
PY=~/envs/sasha-seg/bin/python
M=/var/tmp/venkatavks/c16_roll; O=/var/tmp/venkatavks/real_train; S=/data1/namanm/dataset/camelyon16/images/all
CK=/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt
export CAMELYON16_MIRROR=$M
mkdir -p $M $O
$PY - > $O/names.txt <<'EOF'
import os, numpy as np
d = os.environ['RLSCREEN_ROOT'] + '/units_kaiko_vitl14'
for f in sorted(os.listdir(d)):
    if f.endswith('.npz'):
        z = np.load(f'{d}/{f}')
        if str(z['split']) == 'train' and bool(z['trust']) and z['tissue'].any():
            print(f[:-4])
EOF
wc -l < $O/names.txt
todo() { while read n; do [ -f $O/$n.npz ] || echo $n; done < $O/names.txt; }
while :; do
  b=$(todo | head -8)
  [ -z "$b" ] && break
  rm -f $M/*.tif
  echo $b | tr ' ' '\n' | xargs -P 4 -I{} sh -c "cp $S/{}.tif $M/ 2>/dev/null || cp /data2/venkatavks/c16_extra/{}.tif $M/"
  $PY -m experiments.tokhead.infer --ckpt $CK --out_dir $O --slides $b >> $O/infer.log 2>&1 || { echo infer failed; break; }
  rm -f $M/*.tif
done
rm -f $M/*.tif
echo "train maps done"
