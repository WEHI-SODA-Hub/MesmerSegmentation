# Container
Mesmer container is located at `/vast/projects/SODA/MibiSegmentation/deepcell_mesmer_0.4.1_noentry.sif`

# Parameters

```
usage: run_app.py mesmer [-h] [--output-directory OUTPUT_DIRECTORY] [--output-name OUTPUT_NAME] [-L {DEBUG,INFO,WARN,ERROR,CRITICAL}] [--squeeze] --nuclear-image NUCLEAR_PATH
                         [--nuclear-channel NUCLEAR_CHANNEL [NUCLEAR_CHANNEL ...]] [--membrane-image MEMBRANE_PATH] [--membrane-channel MEMBRANE_CHANNEL [MEMBRANE_CHANNEL ...]]
                         [--image-mpp IMAGE_MPP] [--batch-size BATCH_SIZE] [--compartment {nuclear,whole-cell,both}]

optional arguments:
  -h, --help            show this help message and exit
  --output-directory OUTPUT_DIRECTORY, -o OUTPUT_DIRECTORY
                        Directory where application outputs are saved.
  --output-name OUTPUT_NAME, -f OUTPUT_NAME
                        Name of output file.
  -L {DEBUG,INFO,WARN,ERROR,CRITICAL}, --log-level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        Only log the given level and above.
  --squeeze             Squeeze the output tensor before saving.
  --nuclear-image NUCLEAR_PATH, -n NUCLEAR_PATH
                        REQUIRED: Path to 2D single channel TIF file.
  --nuclear-channel NUCLEAR_CHANNEL [NUCLEAR_CHANNEL ...], -nc NUCLEAR_CHANNEL [NUCLEAR_CHANNEL ...]
                        Channel(s) to use of the nuclear image. If more than one channel is passed, all channels will be summed.
  --membrane-image MEMBRANE_PATH, -m MEMBRANE_PATH
                        Path to 2D single channel TIF file. Optional. If not provided, membrane channel input to network is blank.
  --membrane-channel MEMBRANE_CHANNEL [MEMBRANE_CHANNEL ...], -mc MEMBRANE_CHANNEL [MEMBRANE_CHANNEL ...]
                        Channel(s) to use of the membrane image. If more than one channel is passed, all channels will be summed.
  --image-mpp IMAGE_MPP
                        Input image resolution in microns-per-pixel. Default value of 0.5 corresponds to a 20x zoom.
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for `model.predict`.
  --compartment {nuclear,whole-cell,both}, -c {nuclear,whole-cell,both}
                        The cellular compartment to segment.
```
