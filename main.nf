include { DEEPCELL_MESMER } from '../modules/nf-core/deepcell/mesmer/main'                                                                                  

workflow {
    input_images = channel.fromPath(params.input)
    DEEPCELL_MESMER()
}
