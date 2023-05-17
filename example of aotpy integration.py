import turlib.turlib as tp

path_G = 'G_ordered.csv'
path = 'naomi_system_testing.fits'

vector_tp, turbulence_data = tp.reader_aotpy(path, path_G, dimm_data=True)

r0, L0, fitted_ai2 = tp.iterative_estimator(d=vector_tp[0], modes=vector_tp[1], ai2=vector_tp[2],
                                            noise_estimate=vector_tp[3], n_rec_modes=vector_tp[4], m=vector_tp[5],
                                            c_mat=vector_tp[6], full_vector=False, n_iter=5)

print('seeing at zenith [arcsec] ==', tp.seeing_at_zenith(r0, turbulence_data[1]))
print('dimm seeing [arcsec] ==', turbulence_data[0])
print('Difference between seeing estimates [%] ==',
      abs(tp.seeing_at_zenith(r0, turbulence_data[1]) - turbulence_data[0]) * 100 / turbulence_data[0])
