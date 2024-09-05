#!/bin/bash
# Ellis Brown

start=`date +%s`

# Set the target directory to save the data
target_dir="/content/drive/MyDrive/ssd.pruning.pytorch/data"

# Create the target directory if it doesn't exist
if [ ! -d "$target_dir" ]; then
    echo "Creating directory: $target_dir"
    mkdir -p "$target_dir"
fi

# Navigate to the target directory
echo "Navigating to $target_dir ..."
cd "$target_dir"

echo "Downloading VOC2007 trainval ..."
# Download the data.
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

echo "Downloading VOC2007 test data ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar

echo "Removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in $runtime seconds"
# #!/bin/bash
# # Ellis Brown

# start=`date +%s`

# # handle optional download dir
# if [ -z "$1" ]
#   then
#     # navigate to ~/data
#     echo "navigating to ~/data/ ..." 
#     mkdir -p ~/data
#     cd ~/data/
#   else
#     # check if is valid directory
#     if [ ! -d $1 ]; then
#         echo $1 "is not a valid directory"
#         exit 0
#     fi
#     echo "navigating to" $1 "..."
#     cd $1
# fi

# echo "Downloading VOC2007 trainval ..."
# # Download the data.
# curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
# echo "Downloading VOC2007 test data ..."
# curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# echo "Done downloading."

# # Extract data
# echo "Extracting trainval ..."
# tar -xvf VOCtrainval_06-Nov-2007.tar
# echo "Extracting test ..."
# tar -xvf VOCtest_06-Nov-2007.tar
# echo "removing tars ..."
# rm VOCtrainval_06-Nov-2007.tar
# rm VOCtest_06-Nov-2007.tar

# end=`date +%s`
# runtime=$((end-start))

# echo "Completed in" $runtime "seconds"