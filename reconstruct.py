from brainbuilder.reconstruct import reconstruct


reconstruct('hemi_info.csv', 
            'chunk_info.csv', 
            'sect_info.csv', 
            [1.2,.8,.4,.2,.1,0.05,0.025,0.01],
            './output/', 
            use_syn = True,
            use_3d_syn_cc = True,
            seg_method = 'identity',
            interp_method = 'volumetric',
            use_3d_align_stage = False,
            )
