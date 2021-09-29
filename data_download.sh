mkdir -p RT_DATA/inputs
mkdir -p RT_DATA/outputs_clear_sky
mkdir -p RT_DATA/outputs_pristine

# Uncomment all lines to download all data (which will take time though :)). 

echo "Downloading metadata & statistics..."
curl https://climart.blob.core.windows.net/climart-dataset/META_INFO.json --output RT_DATA/META_INFO.json
curl https://climart.blob.core.windows.net/climart-dataset/statistics.npz --output RT_DATA/statistics.npz
echo "Done." 

echo "Downloading input files..."
#for x in {1979..1991};do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output RT_DATA/inputs/$x.h5; done
#for x in {1994..2014};do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output RT_DATA/inputs/$x.h5; done
for x in 1997 1998 2014 ;do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output RT_DATA/inputs/$x.h5; done
#for x in 1850 1851 1852 ;do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output RT_DATA/inputs/$x.h5; done
#for x in 2097 2098 2099 ;do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output RT_DATA/inputs/$x.h5; done

echo "Done." 

echo "Downloading clear-sky outputs..."
#for x in {1979..1991};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output RT_DATA/outputs_clear_sky/$x.h5; done
#for x in {1994..2014};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output RT_DATA/outputs_clear_sky/$x.h5; done
#for x in 1850 1851 1852 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output RT_DATA/outputs_clear_sky/$x.h5; done
#for x in 2097 2098 2099 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output RT_DATA/outputs_clear_sky/$x.h5; done
echo "Done." 

echo "Downloading pristine-sky outputs..."
for x in 1997 1998 2014 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output RT_DATA/outputs_pristine/$x.h5; done
#for x in {1979..1991};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output RT_DATA/outputs_pristine/$x.h5; done
#for x in {1994..2014};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output RT_DATA/outputs_pristine/$x.h5; done
#for x in 1850 1851 1852 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output RT_DATA/outputs_pristine/$x.h5; done
#for x in 2097 2098 2099 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output RT_DATA/outputs_pristine/$x.h5; done

echo "Done. Finished downloading all data." 
