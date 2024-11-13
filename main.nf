include { DEEPCELL_MESMER } from './modules/nf-core/deepcell/mesmer/main'

process mesmer {
    conda("")
    input:
        tuple(
            val(metadata),
            path(image)
        )
    shell:
        """#!/usr/bin/env python
        from tifffile import TiffFile
        from deepcell.applications import Mesmer
        """
}

workflow {
    nuclear_images = Channel.fromPath(params.images) | map { image -> [[ name: image.name ], image]} | view
    DEEPCELL_MESMER(nuclear_images, nuclear_images)
}
