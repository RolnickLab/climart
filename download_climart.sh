# Change the directory where the data will be downloaded below
data_dir="ClimART_DATA"
mkdir -p ${data_dir}/inputs
mkdir -p ${data_dir}/outputs_clear_sky
mkdir -p ${data_dir}/outputs_pristine

# Uncomment all lines to download all data (which will take time though :)).

echo "Downloading metadata & statistics..."
curl https://climart.blob.core.windows.net/climart-dataset/META_INFO.json --output ${data_dir}/META_INFO.json
curl https://climart.blob.core.windows.net/climart-dataset/statistics.npz --output ${data_dir}/statistics.npz
curl https://climart.blob.core.windows.net/climart-dataset/areacella_fx_CanESM5.nc --output ${data_dir}/areacella_fx_CanESM5.npz
echo "Done."

echo "Downloading input files..."
for x in {1979..1991};do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output ${data_dir}/inputs/$x.h5; done
for x in {1994..2014};do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output ${data_dir}/inputs/$x.h5; done
for x in 1850 1851 1852 ;do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output ${data_dir}/inputs/$x.h5; done
for x in 2097 2098 2099 ;do  curl https://climart.blob.core.windows.net/climart-dataset/inputs/$x.h5 --output ${data_dir}/inputs/$x.h5; done
echo "Done."

echo "Downloading clear-sky targets..."
for x in {1979..1991};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output ${data_dir}/outputs_clear_sky/$x.h5; done
for x in {1994..2014};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output ${data_dir}/outputs_clear_sky/$x.h5; done
for x in 1850 1851 1852 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output ${data_dir}/outputs_clear_sky/$x.h5; done
for x in 2097 2098 2099 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_clear_sky/$x.h5 --output ${data_dir}/outputs_clear_sky/$x.h5; done
echo "Done."

echo "Downloading pristine-sky targets..."
for x in {1979..1991};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output ${data_dir}/outputs_pristine/$x.h5; done
for x in {1994..2014};do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output ${data_dir}/outputs_pristine/$x.h5; done
for x in 1850 1851 1852 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output ${data_dir}/outputs_pristine/$x.h5; done
for x in 2097 2098 2099 ;do  curl https://climart.blob.core.windows.net/climart-dataset/outputs_pristine/$x.h5 --output ${data_dir}/outputs_pristine/$x.h5; done

echo "Done. Finished downloading ClimART :)"
