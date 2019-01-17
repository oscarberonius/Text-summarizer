rm -rf clean_data1.tar.gz
rm -rf clean_data2.tar.gz

aws s3 cp s3://oscar-multi-doc-summarizer-thesis/clean_data1.tar.gz clean_data1.tar.gz
aws s3 cp s3://oscar-multi-doc-summarizer-thesis/clean_data2.tar.gz clean_data2.tar.gz