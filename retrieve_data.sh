 #!/bin/sh

# Download from the zenodo record for data attached to this manuscript: https://zenodo.org/record/5111607
zenodo_record="https://zenodo.org/api/records/5111607"
files=$(curl $zenodo_record | jq -r ".files[].links.self")

for file in $files
do
    echo "Retrieving $(basename $file)"
    wget $file
done