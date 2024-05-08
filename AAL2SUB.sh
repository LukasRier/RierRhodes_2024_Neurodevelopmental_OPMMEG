#!/bin/bash

if [[ ( $@ == "--help") ||  $@ == "-h" || $# == 0 ]]
then 
		echo " "
		echo "Usage: $0 <projectDirectory> <subjectNumber>"
		echo " "
		echo "projectDirectory...BIDS directory containing /sub-xxx/ses-yyy/anat/sub-xxx_anat.nii"
		echo "subjectNumber......value of xxx above. For example 001,002 etc."
		echo " "
		echo "Needs: projectDirectory/derivatives/sourcespace/sub-xxx/sub-xxx_brain.nii (run fsl brain extraction tool)"
		echo "       projectDirectory/derivatives/sourcespace/AAL_regions_1mm_mas.nii.gz (AAL region masks)"
		echo "       projectDirectory/derivatives/sourcespace/MNI152_T1_1mm_brain.nii.gz (MNI template)"
			exit 0
fi 
export FSLOUTPUTTYPE=NIFTI_GZ
project_dir=$1 #Bids directory containing /sub-001/ses-001/anat/sub-001_anat.nii
sub=$2 #001,002....

ANAT="${project_dir}/derivatives/sourcespace/sub-${sub}/sub-${sub}_brain"
AALregs_sub="${project_dir}/derivatives/sourcespace/sub-${sub}/sub-${sub}_AAL_regions"
#AALcentroids_sub="${project_dir}/derivatives/sourcespace/sub-${sub}/sub-${sub}_AAL_centroids"
ANAT2MNI="${project_dir}/derivatives/sourcespace/sub-${sub}/sub-${sub}_anat2mni"
MNI2ANAT="${project_dir}/derivatives/sourcespace/sub-${sub}/sub-${sub}_mni2anat"
AALregs="${project_dir}/derivatives/sourcespace/AAL_regions_1mm_mas.nii.gz"
#AALcentroids="${project_dir}/derivatives/sourcespace/AAL_centroids_1mm.nii.gz"
MNI_brain="${project_dir}/derivatives/sourcespace/MNI152_T1_1mm_brain.nii.gz"

# Check files exist
if ! [[ -f "$ANAT.nii" || -f "$ANAT.nii.gz" ]]; then
   echo "$ANAT doesn't exist!"
   exit 1
fi
if ! [ -f "$AALregs" ]; then
   echo "$AALregs doesn't exist!"
   exit 1
fi
#if ! [ -f "$AALcentroids" ]; then
#   echo "$AALcentroids doesn't exist!"
#   exit 1
#fi
if ! [ -f "$MNI_brain" ]; then
   echo "$MNI_brain doesn't exist!"
   exit 1
fi
# ensure no nans in volume
fslmaths $ANAT -nan $ANAT
rm $ANAT.nii
# flirt anatomical to MNI saving ANAT2MNI transform
echo "flirt ANAT -> MNI brain"
flirt -in ${ANAT}.nii.gz -ref ${MNI_brain} -out ${ANAT2MNI} -omat ${ANAT2MNI}.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear

#Invert ANAT2MNI transform to get MNI2ANAT transform
echo "converting transform ANAT2MNI -> MNI2ANAT"
convert_xfm -omat ${MNI2ANAT}.mat -inverse ${ANAT2MNI}.mat

#Apply inverted transform to AAL regions and centroids
echo "warping AAL regions to subject space"
flirt -in ${AALregs} -applyxfm -init ${MNI2ANAT}.mat -out ${AALregs_sub} -paddingsize 0.0 -interp trilinear -ref ${ANAT}

#echo "warping centroids to subject space"
#flirt -in ${AALcentroids} -applyxfm -init ${MNI2ANAT}.mat -out ${AALcentroids_sub} -paddingsize 0.0 -interp trilinear -ref ${ANAT}
