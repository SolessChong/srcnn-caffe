TOOLS=../../build/tools
DATA=../../data/SR_NE_ANR/Cropped-g

$TOOLS/convert_imageset.bin -g $DATA/data/ $DATA/data/list_train.txt SR-data 0 leveldb
$TOOLS/convert_imageset.bin -g $DATA/label/ $DATA/label/list_train.txt SR-label 0 leveldb

$TOOLS/convert_imageset.bin -g $DATA/data/ $DATA/data/list_test.txt SR-data-test 0 leveldb
$TOOLS/convert_imageset.bin -g $DATA/label/ $DATA/label/list_test.txt SR-label-test 0 leveldb
