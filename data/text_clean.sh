#! bin/bash
cat practice.json| sed -e 's/岁//'| sed -e 's/(//g' | sed -e 's/)//g' | sed -e 's/\///g'| sed -e 's/ (//g' | sed -e 's/质量QAQC/质量\(QA\/QC\)/g' > train_clean.json
cat test.json | sed -e 's/岁//' | sed -e 's/(//g' | sed -e 's/)//g' | sed -e 's/\///g'  | sed -e 's/ (//g' | sed -e 's/质量QAQC/质量\(QA\/QC\)/g' > test_clean.json

