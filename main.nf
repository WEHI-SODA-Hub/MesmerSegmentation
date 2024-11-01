include { DEEPCELL_MESMER } from './modules/nf-core/deepcell/mesmer/main'

workflow {
    input_images = channel.fromPath(params.input) | map { image -> [{ name: image.name }, image]} | view
    membrane = channel.from([{}, null])
    DEEPCELL_MESMER(input_images, membrane)
}
