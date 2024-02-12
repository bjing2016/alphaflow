for name in $(cat `dirname $0`/../splits/atlas.csv | grep -v name | awk -F ',' {'print $1'}); do
    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip
    mkdir ${name}
    unzip ${name}_protein.zip -d ${name}
    rm ${name}_protein.zip
done